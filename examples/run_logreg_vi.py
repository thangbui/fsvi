import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt
import torch.distributions as dists
import numpy as np

from vi import bnn_vi, likelihoods
from sklearn.datasets import make_moons

import pdb
import palettable


class RegDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.targets = y
        self.len = x.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.len


def run_exp(
    no_train_points=100,
    no_test_points=50,
    seed=0,
    no_train_samples=5,
    no_eval_samples=50,
    no_epochs=5000,
    learning_rate=0.05,
    batch_size=50,
    plot=True,
    act_func="relu",
    space="param",
    no_context_samples=50,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()

    # create dataset
    x_train, y_train = make_moons(no_train_points, noise=0.1, random_state=0)
    x_test, y_test = make_moons(no_test_points, noise=0.1, random_state=0)
    train_data = RegDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    test_data = RegDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

    if space.startswith("function"):
        x_context_bounds = [-10, 10]
        no_context_points = 100
        # x_context = get_context_points(x_train, 20, [-3, 4])
        # x_context = x_context.cuda() if use_cuda else x_context
    else:
        x_context_bounds = None
        no_context_points = None

    # create a model
    likelihood = likelihoods.BinarySigmoid()
    if act_func == "relu":
        act = torch.relu
    elif act_func == "tanh":
        act = torch.tanh
    elif act_func == "sigmoid":
        act = torch.sigmoid

    model = bnn_vi.MLP(
        [x_train.shape[1], 1],
        likelihood,
        act,
        use_cuda=use_cuda,
    )
    if use_cuda:
        model = model.cuda()

    def callback_fn(net, epoch):
        if epoch % 1000 == 0:
            net.eval()
            logits = net.predict(
                test_loader, no_samples=no_eval_samples, local=True
            )
            logs = torch.nn.functional.logsigmoid(logits.detach().cpu())
            logprobs = torch.logsumexp(logs, dim=0) - np.log(logs.shape[0])
            probs = torch.exp(logprobs)
            predicted = (probs > 0.5) * 1.0
            y = test_loader.dataset.targets
            acc = predicted.squeeze().eq(y).sum().item()
            ll = dists.Bernoulli(probs.squeeze()).log_prob(y).sum().item()
            net.train()
            acc /= len(test_data)
            ll /= len(test_data)
            print(f"epoch={epoch}; test acc={100 * acc:.2f}%; LL={ll:.4f}")


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = model.train_vi(
        train_loader,
        optimizer,
        no_epochs=no_epochs,
        no_samples=no_train_samples,
        print_cadence=5,
        callback=callback_fn,
        kl_weight=1.0,
        space=space,
        x_context_bounds=x_context_bounds,
        no_context_points=no_context_points,
        no_context_samples=no_context_samples,
    )
    # lml_estimate = model.compute_vi_objective(train_loader, no_train_samples, space, x_context, no_context_samples)

    def setup_grid(X, h=0.05, buffer=2.0):
        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - buffer, X[:, 0].max() + buffer
        y_min, y_max = X[:, 1].min() - buffer, X[:, 1].max() + buffer
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
        )
        grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
        return grid, xx, yy

    if plot:
        grid, xx, yy = setup_grid(x_train)
        no_samples = no_eval_samples
        model.eval()
        plot_data = RegDataset(torch.Tensor(grid), torch.Tensor(grid))
        plot_loader = DataLoader(dataset=plot_data, batch_size=batch_size)
        logits = model.predict(plot_loader, no_samples, local=True)
        logs = torch.nn.functional.logsigmoid(logits.detach().cpu())
        logprobs = torch.logsumexp(logs, dim=0) - np.log(logs.shape[0])
        probs = torch.exp(logprobs).reshape(xx.shape)

        ms = 100
        clevels = 9
        title_fs = 24
        ticks_fs = 18
        cmap_fg = plt.cm.coolwarm
        cmap_bg = palettable.colorbrewer.diverging.RdBu_5_r.get_mpl_colormap()

        fig, ax = plt.subplots(1, 1)
        c = ax.contourf(
            xx,
            yy,
            probs,
            cmap=cmap_bg,
            alpha=0.7,
            levels=clevels,
            antialiased=True,
        )
        ax.scatter(
            x_train[y_train == 0, 0],
            x_train[y_train == 0, 1],
            color=cmap_fg(0),
            s=ms,
            lw=1.5,
            edgecolors="black",
            marker="s",
        )
        ax.scatter(
            x_train[y_train == 1, 0],
            x_train[y_train == 1, 1],
            color=cmap_fg(0.999),
            s=ms,
            lw=1.5,
            edgecolors="black",
        )
        fname = (
            "/tmp/dataset_%s_no_train_%d_no_test_%d_logreg_act_%s"
            % (
                "moon",
                no_train_points,
                no_test_points,
                act_func,
            )
        )
        fname += "_seed_%d_method_%s_space_%s" % (
            seed,
            "mfvi",
            space,
        )
        plt.savefig(
            fname + "_prediction.pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

        plt.figure()
        plt.plot(losses)

        plt.savefig(
            fname + "_loss.pdf",
            bbox_inches="tight",
            pad_inches=0,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(run_exp)

# python run_logreg_vi.py
