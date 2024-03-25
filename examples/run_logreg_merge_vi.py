import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt
import torch.distributions as dists
import numpy as np

from vi import bnn_vi, likelihoods
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from reg_utils import get_reg_data, RegDataset, homogeneous_split, inhomogeneous_split_logreg
from data.split_data import homogenous1, inhomogenous1

import pdb
import palettable

def compute_mse_and_ll(truth, pred, loss_fn, log_w=None):
    if log_w is not None:
        pred_mod = pred * log_w.exp().unsqueeze(-1).unsqueeze(-1)
    else:
        pred_mod = pred
    mse = torch.mean((pred_mod - truth) ** 2)
    neg_loss = -loss_fn(pred, truth, reduction=False, use_cuda=False)
    if log_w is not None:
        neg_loss += log_w.unsqueeze(-1)
    ll = torch.logsumexp(neg_loss, 0).mean()
    if log_w is None:
        ll -= np.log(neg_loss.shape[0])
    return mse, ll

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
    no_epochs_client=10,
    no_epochs_server=100,
    learning_rate=0.005,
    batch_size=50,
    plot=True,
    act_func="relu",
    space_client="param",
    no_context_samples=50,
    homogeneous=True,
    no_clients=1,
    dataset="iris",
    log_fre=10,
    client_size_factor=0,
    class_balance_factor=0,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()

    split_function = homogeneous_split if homogeneous else inhomogeneous_split_logreg

    # create dataset
    if dataset == "iris":
        iris = load_iris()
        X = iris.data[:, :2]
        y = (iris.target == 0).astype(int)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        x_train = x_train.cuda() if use_cuda else x_train
        y_train = y_train.cuda() if use_cuda else y_train
        x_test = x_test.cuda() if use_cuda else x_test
        y_test = y_test.cuda() if use_cuda else y_test
        train_data = RegDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
        test_data = RegDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size)
    elif dataset in ["adult", "bank", "credit"]:
        def load(path):
            return np.load(path)
        X = load(f"/Users/wangmengqi/Desktop/research/codes/test/fsvi/examples/dataset/%s/x.npy" % dataset)
        y = load(f"/Users/wangmengqi/Desktop/research/codes/test/fsvi/examples/dataset/%s/y.npy"% dataset).squeeze(-1)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        x_train = x_train.cuda() if use_cuda else x_train
        y_train = y_train.cuda() if use_cuda else y_train
        x_test = x_test.cuda() if use_cuda else x_test
        y_test = y_test.cuda() if use_cuda else y_test
        train_data = RegDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
        test_data = RegDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

    input_dim = x_train.shape[1]
    output_dim = 1

    if dataset == "iris":
        client_data = split_function(x_train, y_train, no_clients)
    elif dataset in ["adult", "bank", "credit"]:
        if homogeneous:
            client_data = homogenous1(x_train, y_train, no_clients, seed)
        else:
            client_data = inhomogenous1(x_train, y_train, no_clients, client_size_factor, class_balance_factor, seed)
    client_list = []

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
    def callback_fn(net, epoch):
        if epoch % log_fre == 0:
            acc, ll = eval_res(net)
            print(f"epoch={epoch}; test acc={100 * acc:.2f}%; LL={ll:.4f}")

    def eval_res(net):
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
        return acc, ll

    for k in range(no_clients):
        print("training client %d" % k)
        x_k, y_k = client_data[k]['x'], client_data[k]['y']
        train_data_k = RegDataset(torch.Tensor(x_k), torch.Tensor(y_k))
        train_loader_k = DataLoader(dataset=train_data_k, batch_size=batch_size)

        if space_client.startswith("function"):
            x_context_bounds = [-10, 10]
            no_context_points = 50
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
            [input_dim, output_dim],
            likelihood,
            act,
            use_cuda=use_cuda,
        )
        if use_cuda:
            model = model.cuda()

        # def callback_fn(net, epoch):
        #     if epoch % log_fre == 0:
        #         net.eval()
        #         logits = net.predict(
        #             test_loader, no_samples=no_eval_samples, local=True
        #         )
        #         logs = torch.nn.functional.logsigmoid(logits.detach().cpu())
        #         logprobs = torch.logsumexp(logs, dim=0) - np.log(logs.shape[0])
        #         probs = torch.exp(logprobs)
        #         predicted = (probs > 0.5) * 1.0
        #         y = test_loader.dataset.targets
        #         acc = predicted.squeeze().eq(y).sum().item()
        #         ll = dists.Bernoulli(probs.squeeze()).log_prob(y).sum().item()
        #         net.train()
        #         acc /= len(test_data)
        #         ll /= len(test_data)
        #         print(f"epoch={epoch}; test acc={100 * acc:.2f}%; LL={ll:.4f}")


        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        losses = model.train_vi(
            train_loader_k,
            optimizer,
            no_epochs=no_epochs_client,
            no_samples=no_train_samples,
            print_cadence=5,
            callback=callback_fn,
            kl_weight=1.0,
            space=space_client,
            x_context_bounds=x_context_bounds,
            no_context_points=no_context_points,
            no_context_samples=no_context_samples,
        )
        acc, ll = eval_res(model)
        print(f"client {k} {space_client} space result: test acc={100 * acc:.2f}%; LL={ll:.4f}")

        fname = "/tmp/logreg_dataset_%s_homogenous_%s_network_size_%s_logreg_act_%s" % (
            dataset,
            homogeneous,
            1,
            act_func
        )
        fname += (
                "_seed_%d_space_client_%s_client_id_%d"
                % (
                    seed,
                    space_client,
                    k
                )
        )
        plt.figure()
        plt.plot(losses)

        plt.savefig(
            fname + "_loss.pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

        if plot:
            grid, xx, yy = setup_grid(x_k)
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
                x_k[y_k == 0, 0],
                x_k[y_k == 0, 1],
                color=cmap_fg(0),
                s=ms,
                lw=1.5,
                edgecolors="black",
                marker="s",
            )
            ax.scatter(
                x_k[y_k == 1, 0],
                x_k[y_k == 1, 1],
                color=cmap_fg(0.999),
                s=ms,
                lw=1.5,
                edgecolors="black",
            )

            plt.savefig(
                fname + "_prediction.pdf",
                bbox_inches="tight",
                pad_inches=0,
            )

        client_list.append(model)

    for space_server in ["param", "function_sampling"]:
        print(f"merging {space_server}...")

        if space_server.startswith("function"):
            x_context_bounds = [-10, 10]
            no_context_points = 50
        else:
            x_context_bounds = None
            no_context_points = None
        # create a model
        likelihood = likelihoods.BinarySigmoid()

        model = bnn_vi.MLP(
            [input_dim, output_dim],
            likelihood,
            act,
            use_cuda=use_cuda,
        )
        if use_cuda:
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        losses = model.merge_vi(
            client_list,
            optimizer,
            no_epochs=no_epochs_server,
            print_cadence=5,
            callback=callback_fn,
            space=space_server,
            x_context_bounds=x_context_bounds,
            no_context_points=no_context_points,
            no_context_samples=no_context_samples,
        )
        # print results
        acc, ll = eval_res(model)
        print(f"{space_server} space result: test acc={100 * acc:.2f}%; LL={ll:.4f}")

        fname = (
                "/tmp/logreg_server_dataset_%s_no_train_%d_no_test_%d_logreg_act_%s_no_client%d"
                % (
                    dataset,
                    no_train_points,
                    no_test_points,
                    act_func,
                    no_clients,
                )
        )
        fname += "_seed_%d_method_%s_space_server_%s_space_client_%s" % (
            seed,
            "mfvi",
            space_server,
            space_client
        )
        if space_server.startswith("function"):
            plt.figure()
            plt.plot(losses)

            plt.savefig(
                fname + "_function_loss.pdf",
                bbox_inches="tight",
                pad_inches=0,
            )

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

            # fig, ax = plt.subplots(1, 1)
            plt.figure()
            c = plt.contourf(
                xx,
                yy,
                probs,
                cmap=cmap_bg,
                alpha=0.7,
                levels=clevels,
                antialiased=True,
            )
            plt.scatter(
                x_train[y_train == 0, 0],
                x_train[y_train == 0, 1],
                color=cmap_fg(0),
                s=ms,
                lw=1.5,
                edgecolors="black",
                marker="s",
            )
            plt.scatter(
                x_train[y_train == 1, 0],
                x_train[y_train == 1, 1],
                color=cmap_fg(0.999),
                s=ms,
                lw=1.5,
                edgecolors="black",
            )

            plt.savefig(
                fname + "_prediction.pdf",
                bbox_inches="tight",
                pad_inches=0,
            )

            if space_server.startswith("function"):
                plt.figure()
                plt.plot(losses)

                plt.savefig(
                    fname + "_function_loss.pdf",
                    bbox_inches="tight",
                    pad_inches=0,
                )

if __name__ == "__main__":
    import fire

    fire.Fire(run_exp)

# python run_logreg_merge_vi.py --homogeneous=False --space_client="function_sampling" --no_clients=2 --dataset=iris --no_epochs_client=1000 --no_epochs_server=1000 --log_fre=500 --plot=True
# python run_logreg_merge_vi.py --homogeneous=False --space_client="function_sampling" --no_clients=2 --dataset=adult
# python run_logreg_merge_vi.py --homogeneous=False --space_client="function_sampling" --no_clients=2 --dataset=adult --no_epochs_client=20 --no_epochs_server=200 --log_fre=10 --plot=False
# python run_logreg_merge_vi.py --homogeneous=False --space_client="function_sampling" --no_clients=4 --dataset=adult --no_epochs_client=50 --no_epochs_server=200 --log_fre=10 --plot=False --client_size_factor=0 --class_balance_factor=0.6
# python run_logreg_merge_vi.py --homogeneous=False --space_client="param" --no_clients=4 --dataset=adult --no_epochs_client=50 --no_epochs_server=200 --log_fre=10 --plot=False --client_size_factor=0 --class_balance_factor=0.6
# python run_logreg_merge_vi.py --homogeneous=False --space_client="param" --no_clients=4 --dataset=credit --no_epochs_client=50 --no_epochs_server=200 --log_fre=10 --plot=False --client_size_factor=0 --class_balance_factor=0.6