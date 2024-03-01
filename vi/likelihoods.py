import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import pdb


class GaussianLikelihood(nn.Module):
    """
    Independent multi-output Gaussian likelihood.
    """

    def __init__(self, out_size, noise_std, use_cuda=False):
        super().__init__()
        self.noise_std = noise_std * torch.ones(out_size)
        if use_cuda:
            self.noise_std = self.noise_std.cuda()
        self.use_cuda = use_cuda

    def forward(self, mu):
        """
        Arguments:
            mu: no_samples x batch_size x out_size

        Returns:
            observation mean and variance

            obs_mu: no_samples x batch_size x out_size
            obs_var: no_samples x batch_size x out_size
        """
        obs_mu = mu
        zero_vec = torch.zeros_like(obs_mu)
        if self.use_cuda:
            zero_vec = zero_vec.cuda()
        obs_var = zero_vec + torch.square(self.noise_std)
        return obs_mu, obs_var

    def loss(self, f_samples, y, reduction=True, use_cuda=True):
        """
        Arguments:
            f_samples: no_samples x batch_size x out_size
            y: batch_size x out_size

        Returns:
            Total loss scalar value.
        """
        obs_mean, obs_var = self(f_samples)

        if not use_cuda:
            obs_mean = obs_mean.cpu()
            obs_var = obs_var.cpu()

        y_dist = dist.Normal(obs_mean, obs_var.sqrt())
        log_prob = y_dist.log_prob(y)

        nll = -log_prob.sum(dim=-1)
        if reduction:
            nll = nll.sum(-1).mean()
        return nll

    def predict(self, f_samples):
        return f_samples


class MulticlassSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f_samples):
        """
        Arguments:
            f_samples: no_samples x batch_size x out_size

        Returns:
            Predictions for function samples.

            y_f: no_samples x batch_size x out_size
        """
        y_f = F.log_softmax(f_samples, dim=-1)

        return y_f

    def loss(self, f_samples, y, reduction=True):
        """
        Arguments:
            f_samples: no_samples x batch_size x out_size
            y: batch_size

        Returns:
            Total loss scalar value.
        """
        if len(f_samples.shape) == 2:
            f_samples = f_samples.unsqueeze(0)
        no_samples, batch_size = f_samples.shape[0], f_samples.shape[1]
        y = y.unsqueeze(0).expand(no_samples, -1)
        ls = self(f_samples).permute([0, 2, 1])
        lik_loss = F.nll_loss(ls, y, reduction="none")
        if reduction:
            lik_loss = lik_loss.sum(dim=-1)
        return lik_loss

    def predict(self, f_samples):
        """
        Arguments:
            f_samples: no_samples x batch_size x out_size

        Returns
            output probabilities no_samples x batch_size x out_size
        """
        y_f = self(f_samples)
        probs = y_f.exp()
        return probs


class BinarySigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f_samples):
        """
        Arguments:
            f_samples: no_samples x batch_size x out_size

        Returns:
            Predictions for function samples.

            y_f: no_samples x batch_size x out_size
        """
        y_f = F.logsigmoid(f_samples)

        return y_f

    def loss(self, f_samples, y, reduction=True):
        """
        Arguments:
            f_samples: no_samples x batch_size x out_size
            y: batch_size

        Returns:
            Total loss scalar value.
        """
        if len(f_samples.shape) == 2:
            f_samples = f_samples.unsqueeze(0)
        no_samples, batch_size = f_samples.shape[0], f_samples.shape[1]
        y = y.unsqueeze(0).expand(no_samples, -1)
        if len(y.shape) == 2:
            y = y.unsqueeze(1)
        ls = f_samples.permute([0, 2, 1])
        lik_loss = F.binary_cross_entropy_with_logits(ls, y, reduction="none")
        if reduction:
            lik_loss = lik_loss.sum(dim=-1)
        return lik_loss

    def predict(self, f_samples):
        """
        Arguments:
            f_samples: no_samples x batch_size x out_size

        Returns
            output probabilities no_samples x batch_size x out_size
        """
        y_f = self(f_samples)
        probs = y_f.exp()
        return probs
