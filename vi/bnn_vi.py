from random import sample
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, MultivariateNormal
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F

from tqdm.auto import tqdm
import pdb
import math
import copy
import numpy as np
from abc import ABC, abstractmethod

from .utils import StableSqrt, get_context_points


class VariationalLayer(nn.Module, ABC):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def compute_variational_kl_term(self):
        q_post, q_mean, q_var = self.get_post_params()
        p_prior = self.prior_dist
        return kl_divergence(q_post, p_prior).sum()


class VariationalLinearLayer(VariationalLayer):
    def forward(self, inputs, no_samples, local=True):
        if self.bias:
            onevec = torch.ones(inputs.shape[0], inputs.shape[1], 1)
            if self.use_cuda:
                onevec = onevec.cuda()
            inputs = torch.cat([inputs, onevec], 2)
        if local:
            return self._forward_local(inputs, no_samples)
        else:
            return self._forward_nonlocal(inputs, no_samples)

    def _forward_local(self, inputs, no_samples):
        q_post, q_mean, q_var = self.get_post_params()
        mz = torch.einsum("kni,io->kno", inputs, q_mean)
        vz = torch.einsum("kni,io->kno", inputs**2, q_var)
        eps = torch.empty(vz.size(), device=vz.device).normal_(0.0, 1.0)
        z_samples = eps * StableSqrt.apply(vz) + mz
        return z_samples

    def _forward_nonlocal(self, inputs, no_samples):
        params = self.get_weight_samples(no_samples)
        z_samples = torch.einsum("kni,kio->kno", inputs, params)
        return z_samples

    def forward_custom(self, inputs, no_samples, dist):
        if self.bias:
            onevec = torch.ones(inputs.shape[0], inputs.shape[1], 1)
            if self.use_cuda:
                onevec = onevec.cuda()
            inputs = torch.cat([inputs, onevec], 2)
        params = self.get_weight_samples(no_samples, dist)
        z_samples = torch.einsum("kni,kio->kno", inputs, params)
        return z_samples

    def extra_repr(self) -> str:
        s = "in_features={}, out_features={}".format(self.size[0], self.size[1])
        return s


class GaussianLayer(nn.Module):
    def __init__(self, size, prior_mean=0.0, prior_std=1.0, use_cuda=False):
        super().__init__()
        self.size = S = torch.Size(size)
        self.use_cuda = use_cuda

        # prior
        prior_mean = torch.ones(S) * prior_mean
        prior_std = torch.ones(S) * prior_std
        if use_cuda:
            prior_mean = prior_mean.cuda()
            prior_std = prior_std.cuda()
        self.prior_dist = Independent(
            Normal(loc=prior_mean, scale=prior_std), 2
        )

        # variational parameters
        self.q_mean = nn.Parameter(torch.Tensor(S).normal_(0.0, 0.1))
        self.q_log_std = nn.Parameter(
            torch.log(torch.Tensor([0.01])) * torch.ones(S)
        )

    def get_post_params(self):
        q_mean = self.q_mean
        q_std = torch.exp(self.q_log_std)
        q_dist = Independent(Normal(loc=q_mean, scale=q_std), 2)
        return q_dist, q_mean, q_std**2

    def get_weight_samples(self, no_samples=1, dist="posterior"):
        if dist == "posterior":
            q, _, _ = self.get_post_params()
        elif dist == "prior":
            q = self.prior_dist
        params = q.rsample(torch.Size([no_samples]))
        if self.use_cuda:
            params = params.cuda()
        return params

    def merge_param(self, layers):
        no_clients = len(layers)
        p_prec = 1.0 / self.prior_dist.variance
        p_prec_mean = self.prior_dist.mean * p_prec
        q_mean_list = [layer.q_mean for layer in layers]
        q_prec_list = [1.0 / torch.exp(2 * layer.q_log_std) for layer in layers]
        prec_sum = 0
        prec_mean_sum = 0
        for i, (mean, prec) in enumerate(zip(q_mean_list, q_prec_list)):
            prec_sum += prec
            prec_mean_sum += prec * mean
        prec_sum += (1 - no_clients) * p_prec
        prec_mean_sum += (1 - no_clients) * p_prec_mean
        mean = prec_mean_sum / prec_sum
        var = 1.0 / prec_sum
        self.q_mean.data = mean.data
        self.q_log_std.data = torch.log(var.data) / 2



class GaussianLinear(VariationalLinearLayer, GaussianLayer):
    def __init__(
        self, size, prior_mean=0.0, prior_std=1.0, use_cuda=False, bias=True
    ):
        VariationalLinearLayer.__init__(self)
        din = size[0]
        dout = size[1]
        if bias:
            din = din + 1
        sizeb = [din, dout]
        self.bias = bias
        GaussianLayer.__init__(self, sizeb, prior_mean, prior_std, use_cuda)


class MLP(nn.Module):
    def __init__(
        self,
        network_size,
        likelihood,
        act_func,
        use_cuda=False,
    ):
        nn.Module.__init__(self)
        self.no_layers = len(network_size) - 1
        self.network_size = S = network_size

        self.layers = nn.ModuleList()
        for layer_idx, (i, o) in enumerate(zip(S[:-1], S[1:])):
            layer = GaussianLinear([i, o], use_cuda=use_cuda)
            self.layers.append(layer)
        self.likelihood = likelihood
        self.act_func = act_func
        self.use_cuda = use_cuda

    def compute_ps_kl_term(self):
        kl = 0
        for layer in self.layers:
            kl += layer.compute_variational_kl_term()
        return kl

    def forward(self, x, no_samples, local):
        inputs = x
        for i in range(self.no_layers):
            a = self.layers[i].forward(inputs, no_samples, local)
            inputs = self.act_func(a)
        return a

    def forward_custom(self, x, no_samples, dist):
        inputs = x
        for i in range(self.no_layers):
            a = self.layers[i].forward_custom(inputs, no_samples, dist)
            inputs = self.act_func(a)
        return a

    def loss_mfvi_ps(self, x, y, no_samples, local=True):
        # compute analytic kl term
        kl_term = self.compute_ps_kl_term()
        # and log likelihood term
        x = x.unsqueeze(0).repeat([no_samples, 1, 1])
        outputs = self.forward(x, no_samples, local=True)
        lik = -self.likelihood.loss(outputs, y, reduction=True).mean()
        return kl_term, lik

    def loss_mfvi_fs_sampling(
        self, x, y, no_samples, x_context, no_context_samples, local=True
    ):
        # compute analytic kl term
        x_context = x_context.unsqueeze(0).repeat([no_context_samples, 1, 1])
        q_f_posterior = self.get_qf_sampling(
            x_context, no_context_samples, "posterior"
        )
        q_f_prior = self.get_qf_sampling(x_context, no_context_samples, "prior")
        kl_term = kl_divergence(q_f_posterior, q_f_prior)
        # and log likelihood term
        x = x.unsqueeze(0).repeat([no_samples, 1, 1])
        outputs = self.forward(x, no_samples, local=True)
        lik = -self.likelihood.loss(outputs, y, reduction=True).mean()
        return kl_term, lik

    def compute_vi_objective(
        self, loader, no_samples, space, x_context=None, no_context_samples=100
    ):
        loss = 0
        no_batches = len(loader)
        Nall = len(loader.dataset)
        for x_batch, y_batch in loader:
            if self.use_cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            n_batch = x_batch.shape[0]
            if space == "param":
                kl, lik = self.loss_mfvi_ps(x_batch, y_batch, no_samples)
            elif space == "function_sampling":
                kl, lik = self.loss_mfvi_fs_sampling(
                    x_batch, y_batch, no_samples, x_context, no_context_samples
                )
            loss += (kl - lik / n_batch * Nall).detach()
        loss /= no_batches
        return loss.detach()

    def get_weight_samples(self, no_samples=1):
        weights = []
        with torch.no_grad():
            for layer in self.layers:
                weight = layer.get_weight_samples(no_samples).detach()
                if no_samples == 1:
                    weight = weight.squeeze(0)
                weights.append(weight)
        return weights

    def train_vi(
        self,
        loader,
        optimizer,
        no_samples=5,
        no_epochs=500,
        print_cadence=1,
        callback=None,
        kl_weight=1.0,
        space="param",
        x_context_bounds=None,
        no_context_points=None,
        no_context_samples=50,
    ):
        Nall = len(loader.dataset)
        count = 0
        pbar = tqdm(range(no_epochs))
        losses = []
        for e in pbar:
            for x_batch, y_batch in tqdm(loader, leave=False):
                optimizer.zero_grad()
                shape = x_batch.shape
                n_batch = shape[0]
                if len(shape) > 2:
                    x_batch = x_batch.reshape([n_batch, -1])
                if self.use_cuda:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                if space == "param":
                    kl, lik = self.loss_mfvi_ps(
                        x_batch, y_batch, no_samples, local=True
                    )
                elif space == "function_sampling":
                    x_context = get_context_points(
                        x_batch.shape[-1], no_context_points, x_context_bounds
                    )
                    x_context = x_context.cuda() if self.use_cuda else x_context
                    kl, lik = self.loss_mfvi_fs_sampling(
                        x_batch,
                        y_batch,
                        no_samples,
                        x_context=x_context,
                        no_context_samples=no_context_samples,
                        local=True,
                    )
                loss = kl_weight * kl / Nall - lik / n_batch
                loss.backward()
                optimizer.step()
                if count % print_cadence == 0:
                    pbar.set_postfix(
                        {
                            "epoch": e,
                            "total": no_epochs,
                            "loss": loss.item(),
                            "kl": kl.item(),
                            "lik": lik.item(),
                        }
                    )
                count += 1
                losses.append(loss.detach().cpu().numpy())
            if callback is not None:
                callback(self, e)

        return losses

    def predict(self, loader, no_samples, local=True):
        for i, (x_batch, _) in enumerate(tqdm(loader)):
            shape = x_batch.shape
            if len(shape) > 2:
                x_batch = x_batch.reshape([shape[0], -1])
            if self.use_cuda:
                x_batch = x_batch.cuda()
            x_batch = x_batch.unsqueeze(0).repeat([no_samples, 1, 1])
            output_i = self.forward(x_batch, no_samples, local)
            if i == 0:
                outputs = output_i
            else:
                outputs = torch.cat((outputs, output_i), dim=1)
        return outputs

    def get_qf_sampling(self, x, no_samples, dist):
        samples = self.forward_custom(x, no_samples, dist)
        qf_mean = torch.mean(samples, 0)
        samples_diff = samples - qf_mean
        qf_cov = torch.einsum("kil,kjl->ijl", samples_diff, samples_diff)
        qf_cov /= no_samples
        qf_cov += 1e-1 * torch.eye(qf_cov.shape[0]).unsqueeze(2)
        qf_dist = MultivariateNormal(
            loc=qf_mean.permute([1, 0])[0, :],
            covariance_matrix=qf_cov.permute([2, 0, 1])[0, :, :],
        )
        return qf_dist

    def merge_vi(
        self,
        clients,
        optimizer,
        no_epochs=500,
        print_cadence=1,
        callback=None,
        space="param",
        x_context_bounds=None,
        no_context_points=None,
        no_context_samples=50,
    ):
        if space == "param":
            self.merge_param(clients)
        elif space == "function_sampling":
            losses = self.merge_function_sampling(
                clients,
                optimizer,
                no_epochs,
                print_cadence,
                callback,
                space,
                x_context_bounds,
                no_context_points,
                no_context_samples,
            )
            return losses
        
    def merge_param(self, clients):
        no_clients = len(clients)
        for i, layer in enumerate(self.layers):
            layer.merge_param([client.layers[i] for client in clients])

    def merge_function_sampling(
        self,
        clients,
        optimizer,
        no_epochs=500,
        print_cadence=1,
        callback=None,
        space="param",
        x_context_bounds=None,
        no_context_points=None,
        no_context_samples=50,
    ):
        no_clients = len(clients)
        pbar = tqdm(range(no_epochs))
        losses = []
        count = 0
        for e in pbar:
            optimizer.zero_grad()
            dim = self.layers[0].size[0] - 1 if self.layers[0].bias else 0
            x_context = get_context_points(
                dim, no_context_points, x_context_bounds
            )
            x_context = x_context.cuda() if self.use_cuda else x_context
            x_context = x_context.unsqueeze(0).repeat([no_context_samples, 1, 1])

            qf = self.get_qf_sampling(x_context, no_context_samples, "posterior")
            pf = self.get_qf_sampling(x_context, no_context_samples, "prior")
            
            kl_1 = (1 - no_clients) * kl_divergence(qf, pf)
            kl_2 = 0
            for k in range(no_clients):
                qf_k = clients[k].get_qf_sampling(x_context, no_context_samples, "posterior")
                kl_2 += kl_divergence(qf, qf_k)
            loss = (kl_1 + kl_2) / no_context_points
            loss.backward()
            optimizer.step()
            if count % print_cadence == 0:
                pbar.set_postfix(
                    {
                        "epoch": e,
                        "total": no_epochs,
                        "loss": loss.item(),
                        "kl_client": kl_2.item(),
                        "kl_prior": kl_1.item(),
                    }
                )
            count += 1
            losses.append(loss.detach().cpu().numpy())
            if callback is not None:
                callback(self, e)
        return losses
