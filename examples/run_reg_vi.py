import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt
import numpy as np

from vi import bnn_vi, likelihoods
from reg_utils import get_reg_data, RegDataset

import pdb


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


def run_exp(
    no_train_points=100,
    no_test_points=50,
    dataset="sin",
    network_size=[1, 20, 20, 1],
    real_noise_std=0.1,
    eval_noise_std=0.1,
    seed=0,
    no_train_samples=5,
    no_eval_samples=50,
    no_epochs=5000,
    learning_rate=0.01,
    batch_size=50,
    plot=True,
    act_func="tanh",
    space="param",
    no_context_samples=50,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()

    # create dataset
    x_train, y_train = get_reg_data(dataset, no_train_points, real_noise_std)
    x_test, y_test = get_reg_data(dataset, no_test_points, real_noise_std)
    x_train = x_train.cuda() if use_cuda else x_train
    y_train = y_train.cuda() if use_cuda else y_train
    x_test = x_test.cuda() if use_cuda else x_test
    y_test = y_test.cuda() if use_cuda else y_test
    train_data = RegDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    test_data = RegDataset(x_test, y_test)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)
    output_dim = y_train.shape[1]

    if space.startswith("function"):
        x_context_bounds = [-10, 10]
        no_context_points = 100
    else:
        x_context_bounds = None
        no_context_points = None

    # create a model
    likelihood = likelihoods.GaussianLikelihood(
        output_dim, eval_noise_std, use_cuda
    )
    if act_func == "relu":
        act = torch.relu
    elif act_func == "tanh":
        act = torch.tanh
    elif act_func == "sigmoid":
        act = torch.sigmoid

    model = bnn_vi.MLP(
        network_size,
        likelihood,
        act,
        use_cuda=use_cuda,
    )
    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = model.train_vi(
        train_loader,
        optimizer,
        no_epochs=no_epochs,
        no_samples=no_train_samples,
        print_cadence=5,
        kl_weight=1.0,
        callback=None,
        space=space,
        x_context_bounds=x_context_bounds,
        no_context_points=no_context_points,
        no_context_samples=no_context_samples,
    )
    # lml_estimate = model.compute_vi_objective(train_loader, no_train_samples, space, x_context, no_context_samples)
    local = False
    train_pred = (
        model.predict(train_loader, no_eval_samples, local).detach().cpu()
    )
    test_pred = (
        model.predict(test_loader, no_eval_samples, local).detach().cpu()
    )
    train_mse, train_ll = compute_mse_and_ll(
        y_train.cpu(), train_pred, model.cpu().likelihood.loss
    )
    test_mse, test_ll = compute_mse_and_ll(
        y_test.cpu(), test_pred, model.cpu().likelihood.loss
    )

    # print("%s stochastic lower bound %.4f" % ("mfvi", lml_estimate))
    print("train mse %.3f, ll %.3f " % (train_mse, train_ll))
    print("test mse %.3f, ll %.3f " % (test_mse, test_ll))

    fname = "/tmp/dataset_%s_no_train_%d_no_test_%d_network_size_%s_act_%s" % (
        dataset,
        no_train_points,
        no_test_points,
        network_size,
        act_func,
    )
    fname += (
        "_seed_%d_method_%s_space_%s_real_noise_std_%.3f_eval_noise_std_%.3f"
        % (
            seed,
            "mfvi",
            space,
            real_noise_std,
            eval_noise_std,
        )
    )

    if plot:
        Nplot = 200
        xplot = torch.linspace(-6, 6, Nplot).reshape([Nplot, 1])
        plot_dataset = RegDataset(xplot, xplot)
        plot_loader = DataLoader(plot_dataset, batch_size=batch_size)
        plot_samples = model.predict(
            plot_loader, no_samples=no_eval_samples, local=False
        )
        plot_samples = plot_samples.detach().cpu()
        plot_mean = plot_samples.mean(0)
        plot_std = plot_samples.std(0)

        plt.figure()
        plt.plot(xplot, plot_mean, "-k", linewidth=2)
        plt.fill_between(
            xplot[:, 0],
            plot_mean[:, 0]
            + 2 * np.sqrt(plot_std[:, 0] ** 2 + eval_noise_std**2),
            plot_mean[:, 0]
            - 2 * np.sqrt(plot_std[:, 0] ** 2 + eval_noise_std**2),
            color="k",
            alpha=0.3,
        )
        for i in range(no_eval_samples):
            plt.plot(
                xplot,
                plot_samples[i, :, :],
                "-k",
                alpha=0.1,
            )

        plt.plot(x_train.cpu(), y_train.cpu(), "+", color="k", markersize=8)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim([-6, 6])
        plt.ylim([-3.5, 4])
        # plt.show()
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

        # x_context, _ = torch.sort(x_context, dim=0)
        # qf_post = model.get_qf_sampling(x_context.unsqueeze(0).repeat([100, 1, 1]), 100, "posterior")
        # plot_mean = qf_post.loc.detach().cpu()
        # plot_std = torch.diag(qf_post.covariance_matrix).sqrt().detach().cpu()
        # plt.plot(x_context, plot_mean, "-k", linewidth=2)
        # plt.fill_between(
        #     x_context[:, 0],
        #     plot_mean
        #     + 2 * np.sqrt(plot_std ** 2 + eval_noise_std**2),
        #     plot_mean
        #     - 2 * np.sqrt(plot_std ** 2 + eval_noise_std**2),
        #     color="k",
        #     alpha=0.3,
        # )
        # plt.plot(x_train.cpu(), y_train.cpu(), "+", color="k", markersize=8)
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.xlim([-5, 5])
        # plt.ylim([-3.5, 4])

        # qf_post = model.get_qf_sampling(x_context.unsqueeze(0).repeat([100, 1, 1]), 100, "prior")
        # plot_mean = qf_post.loc.detach().cpu()
        # plot_std = torch.diag(qf_post.covariance_matrix).sqrt().detach().cpu()
        # plt.plot(x_context, plot_mean, "-k", linewidth=2)
        # plt.fill_between(
        #     x_context[:, 0],
        #     plot_mean
        #     + 2 * np.sqrt(plot_std ** 2 + eval_noise_std**2),
        #     plot_mean
        #     - 2 * np.sqrt(plot_std ** 2 + eval_noise_std**2),
        #     color="k",
        #     alpha=0.3,
        # )
        # plt.plot(x_train.cpu(), y_train.cpu(), "+", color="k", markersize=8)
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.xlim([-5, 5])
        # plt.ylim([-3.5, 4])

        # plt.show()


if __name__ == "__main__":
    import fire

    fire.Fire(run_exp)

# python run_reg_vi.py --dataset=sin --space=param
# python run_reg_vi.py --dataset=sin --space=function_sampling
# python run_reg_vi.py --dataset=cubic
