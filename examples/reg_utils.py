import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset


# Data splitting functions.
def homogeneous_split(x, y, num_clients=100, seed=42):
    np.random.seed(seed)
    perm = np.random.permutation(len(x))
    client_data = []
    for i in range(num_clients):
        client_idx = perm[i::num_clients]
        client_data.append({"x": x[client_idx], "y": y[client_idx]})

    return client_data


# Data splitting functions.
def inhomogeneous_split(x, y, num_clients=100, seed=42):
    np.random.seed(seed)
    client_data = []
    num_client_data = len(x) // num_clients
    x, idx = torch.sort(x, dim=0)
    y = y[idx[:, 0], :]
    for i in range(num_clients):
        cur_x = x[i * num_client_data:min((i + 1) * num_client_data, len(x))]
        cur_y = y[i * num_client_data:min((i + 1) * num_client_data, len(y))]
        client_data.append({"x": cur_x, "y": cur_y})

    return client_data


# Data splitting functions.
def inhomogeneous_split_logreg(x, y, num_clients=100, seed=42):
    if num_clients == 1:
        return [{"x": x, "y": y}]

    condition1 = (5 / 6 * x[:, 0] - 2 >= x[:, 1]) & (1 / 2 * x[:, 0] + x[:, 1] < 6)
    condition2 = (1 / 2 * x[:, 0] + x[:, 1] >= 6) & (x[:, 1] > 5 / 6 * x[:, 0] - 2)
    client_data = []
    set1 = condition1 | condition2
    set2 = ~set1
    client_data.append({"x": x[set1], "y": y[set1]})
    client_data.append({"x": x[set2], "y": y[set2]})
    print(len(x), len(x[set1]), len(y[set2]))
    return client_data


def get_reg_data(dataset, N, real_noise_std):
    if dataset == "sin":
        x = torch.randn(N, 1) * 2
        x = torch.where(x < 0, x - 4, x)
        x = torch.where(x < 0, x, x + 0.5)
        y = torch.sin(x) + torch.randn(N, 1) * real_noise_std
        # x = (x - x.mean()) / x.std()
        x = (x + 1) / 3.0
    elif dataset == "linear":
        x = torch.randn(N, 1) * 2
        x = torch.where(x < 0, x - 4, x)
        x = torch.where(x < 0, x, x + 0.5)
        y = x / 5 + torch.randn(N, 1) * real_noise_std
        x = (x - x.mean()) / x.std()
    elif dataset == "cubic":
        x = torch.randn(N, 1) * 2
        x = torch.where(x < 0, x - 4, x)
        x = torch.where(x < 0, x, x + 0.5)
        y = x ** 3
        # print(x.mean(), x.std())
        # print(y.mean(), y.std())
        # x = (x - x.mean()) / x.std()
        # y = (y - y.mean()) / y.std()
        x = (x + 1) / 3
        y = (y + 75) / 140
        y += torch.randn(N, 1) * real_noise_std
    elif dataset == "cluster":
        x = torch.cat(
            [
                -1.0 + torch.randn(N // 2, 1) * 0.1,
                1.0 + torch.randn(N // 2, 1) * 0.1,
            ],
            0,
        )
        y = torch.cat(
            [
                -2 + torch.randn(N // 2, 1) * real_noise_std,
                2 + torch.randn(N // 2, 1) * real_noise_std,
            ],
            0,
        )
    else:
        raise Exception("unknown dataset name")
    return x, y


class FourClass2D(Dataset):
    def __init__(self, N_K=50, K=4, X=None, Y=None):
        super().__init__()

        if X is not None:
            self.data, self.targets = X, Y
        else:
            self.data, self.targets = self._init_data(N_K, K)

        self.task_ids = torch.arange(self.targets.size(0))

    def _init_data(self, N_K, K):
        X1 = torch.cat(
            [
                0.8 + 0.4 * torch.randn(N_K, 1),
                1.5 + 0.4 * torch.randn(N_K, 1),
            ],
            dim=-1,
        )
        Y1 = 0 * torch.ones(X1.size(0)).long()

        X2 = torch.cat(
            [
                0.5 + 0.6 * torch.randn(N_K, 1),
                -0.2 - 0.1 * torch.randn(N_K, 1),
            ],
            dim=-1,
        )
        Y2 = 2 * torch.ones(X2.size(0)).long()

        X3 = torch.cat(
            [
                2.5 - 0.1 * torch.randn(N_K, 1),
                1.0 + 0.6 * torch.randn(N_K, 1),
            ],
            dim=-1,
        )
        Y3 = 1 * torch.ones(X3.size(0)).long()

        X4 = torch.distributions.MultivariateNormal(
            torch.Tensor([-0.5, 1.5]),
            covariance_matrix=torch.Tensor([[0.2, 0.1], [0.1, 0.1]]),
        ).sample(torch.Size([N_K]))
        Y4 = 3 * torch.ones(X4.size(0)).long()

        X = torch.cat([X1, X2, X3, X4], dim=0)
        X[:, 1] -= 1
        X[:, 0] -= 0.5

        Y = torch.cat([Y1, Y2, Y3, Y4])

        return X, Y

    def filter_by_class(self, class_list=None):
        if class_list:
            mask = torch.zeros_like(self.targets).bool()
            for c in class_list:
                mask |= self.targets == c
        else:
            mask = torch.ones_like(self.targets).bool()

        self.task_ids = torch.masked_select(
            torch.arange(self.targets.size(0)), mask
        )

    def __getitem__(self, index):
        return (
            self.data[self.task_ids[index]],
            self.targets[self.task_ids[index]],
        )

    def __len__(self):
        return self.task_ids.size(0)


class Banana(Dataset):
    def __init__(self):
        super().__init__()
        self.data, self.targets = self._init_data()
        self.task_ids = torch.arange(self.targets.size(0))

    def _init_data(self):
        X = np.loadtxt("./data/banana_train_x.txt", delimiter=",")
        y = np.loadtxt("./data/banana_train_y.txt", delimiter=",")
        y[y == -1] = 0
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(y).long()
        return X, Y

    def filter_by_input(self, boundary_list=None):
        if boundary_list:
            mask = (self.data[:, 0] < boundary_list[1]) & (
                    self.data[:, 0] > boundary_list[0]
            )
        else:
            mask = torch.ones_like(self.targets).bool()

        self.task_ids = torch.masked_select(
            torch.arange(self.targets.size(0)), mask
        )

    def __getitem__(self, index):
        return (
            self.data[self.task_ids[index]],
            self.targets[self.task_ids[index]],
        )

    def __len__(self):
        return self.task_ids.size(0)


class RegDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
