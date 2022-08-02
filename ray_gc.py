import ray
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data.dataset import random_split
import torchvision

import numpy as np
import os
import random

torch.set_default_dtype(torch.float64)

def seed_torch(seed=1):
    import torch
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def seed_torch_gpu(seed=1):
    seed_torch(seed)
    import torch
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class GC_FR():
    def __init__(self, config):
        self.config = config
        self.n = self.config["n"]
        self.s = self.config["s"]
        self.c = self.config["c"]

    def encode(self, G, rank):
        return sum(G)

    def decode(self, G, rank, start=0):
        y = G[np.remainder(start + 0, len(rank))]
        count = 1
        s = [0 for i in range(self.n // self.c)]
        s[rank[np.remainder(start + 0, len(rank))] // self.c] = 1
        for i in range(1, len(rank)):
            if s[rank[np.remainder(start + i, len(rank))] // self.c] == 0:
                y = y + G[np.remainder(start + i, len(rank))]
                count += 1
                s[rank[np.remainder(start + i, len(rank))] // self.c] = 1
        return y, count * self.c

    def partition(self, partitions, rank):
        return [partitions[i] for i in range(((rank) // self.c) * self.c, ((rank) // self.c + 1) * self.c)]

class IS_GC_FR(GC_FR):
    def __init__(self, config):
        super(IS_GC_FR, self).__init__(config)
        self.c = self.config["c"]

class GC_CR():
    def __init__(self, config):
        self.config = config
        self.n = self.config["n"]
        self.s = self.config["s"]
        self.c = self.config["c"]

        if s != 0:
            a = np.random.rand(self.n - 1, self.n - 1)
            q, _ = np.linalg.qr(a)
            self.H = np.zeros((self.s, self.n))
            self.H[0:self.s,0:(self.n-1)] = q[0:self.s]
            self.H[:, -1] = -np.sum(self.H[:, 0:-1], 1)
            self.B = np.zeros((self.n, self.n))

            for i in range(self.n):
                j = np.remainder(np.arange(i, self.s+i+1), self.n)
                self.B[i,j] = np.append([1], -np.linalg.lstsq(self.H[:, j[1: s+1]], self.H[:, j[0]], rcond=None)[0])

    def encode(self, G, rank):
        if self.s != 0:
            x = self.B[rank][np.remainder(np.arange(rank, rank + self.s + 1), self.n)]
            y = G[0] * x[0]
            for i in range(1, len(x)):
                y += G[i] * x[i]
            return y
        else:
            return G[0]

    def decode(self, G, ranks, start=0):
        if self.s != 0:
            ones = np.ones((1, self.n))[0]
            x = np.linalg.lstsq(self.B[ranks].T, ones, rcond=None)
            y = G[0] * x[0][0]
            for i in range(1, len(x[0])):
                y += G[i] * x[0][i]
            return y, self.n
        else:
            return sum(G), self.n

    def partition(self, partitions, rank):
        return [partitions[i] for i in np.remainder(np.arange(rank, rank + self.c), self.n)]

class IS_GC_CR(IS_GC_FR):
    def partition(self, partitions, rank):
        return GC_CR.partition(self, partitions, rank)

    def decode(self, G, ranks, start=0):
        # a helper function for aisgc cyclic
        def index_max_d(L, c, n):
            if len(L) == 1:
                return L
            L = sorted(L)
            origin = start
            curr = origin
            result = []
            while (L[curr] - L[origin]) % n < c:
                l = [L[curr]]
                head = curr
                nex = (head + 1) % len(L)
                while True:
                    if (L[curr] - L[nex]) % n < c or nex == curr:
                        break
                    if (L[nex] - L[head]) % n >= c:
                        l.append(L[nex])
                        head = nex
                    nex = (nex + 1) % len(L)
                if len(l) > len(result):
                    result = l
                curr = (curr + 1) % len(L)
                if curr == origin:
                    break
            return result

        G0 = {}
        for i in range(len(ranks)):
            G0[ranks[i]] = G[i]
        useful_workers = index_max_d(ranks, self.c, self.n)
        return sum([G0[i] for i in useful_workers]), len(useful_workers) * self.c

class IS_GC_HR(IS_GC_FR):
    def __init__(self, config):
        super(IS_GC_HR, self).__init__(config)
        self.c1 = self.config["c1"]
        self.c2 = self.c - self.c1
        self.g = self.config["g"]
        self.n0 = self.n // self.g

    def partition(self, partitions, rank):
        i = rank % self.n0
        base = rank - i
        index0 = [base + (i + j) % self.n0 for j in range(self.n0 - self.c1, self.n0)]
        index1 = list(np.remainder(np.arange(rank, rank + self.c2), self.n))
        index = index0 + index1
        return [partitions[i] for i in index]

    def decode(self, G, ranks, start=0):
        def conflict_to_right(r1, r2):
            g1 = r1 // self.n0
            g2 = r2 // self.n0
            if g1 == g2:
                if r1 < r2:
                    return True
            if g1 + 1 == g2:
                i1 = r1 % self.n0
                if i1 >= self.n0 - self.c2 + 1 and r2 - r1 < self.c:
                    return True
            if g2 == 0 and g1 == self.g - 1:
                i1 = r1 % self.n0
                if i1 >= self.n0 - self.c2 + 1 and r2 + self.n - r1 < self.c:
                    return True
            return False

        def index_max_d(L, c, n):
            if len(L) == 1:
                return L

            L = sorted(L)
            hashL = {}
            for i in range(len(L)):
                if L[i] // self.n0 in hashL:
                    hashL[L[i] // self.n0].append(i)
                else:
                    hashL[L[i] // self.n0] = [i]
            group = L[start] // self.n0
            result = []
            for curr in hashL[group]:
                l = [L[curr]]
                head = curr
                nex = (head + 1) % len(L)
                while True:
                    if conflict_to_right(L[nex], L[curr]) or nex == curr:
                        break
                    if not conflict_to_right(L[head], L[nex]):
                        l.append(L[nex])
                        head = nex
                    nex = (nex + 1) % len(L)
                if len(l) > len(result):
                    result = l
            return result

        G0 = {}
        for i in range(len(ranks)):
            G0[ranks[i]] = G[i]
        useful_workers = index_max_d(ranks, self.c, self.n)
        return sum([G0[i] for i in useful_workers]), len(useful_workers) * self.c

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        torch.set_default_dtype(torch.float64)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add output layer
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)

from torchvision import models

class ResNet_MNIST(nn.Module):
    def __init__(self):
        super(ResNet_MNIST, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.resnet(x)

class ResNet_CIFAR(nn.Module):
    def __init__(self):
        super(ResNet_CIFAR, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.resnet(x)

class ResNet_Fashion_MNIST(nn.Module):
    def __init__(self):
        super(ResNet_Fashion_MNIST, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.resnet(x)

def train_(batch_idx, model, device, train_loader, optimizer, rank):
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    (data, target) = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return loss, correct

def test(model, device, test_loader):
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(
                dim=1,
                keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return {
        "loss": test_loss,
        "accuracy": 100. * correct / len(test_loader.dataset)
    }

def dataset_creator_MNIST(rank, coder, use_cuda, config):
    n_workers = coder.n
    s_workers = coder.s
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    from filelock import FileLock
    from torchvision import transforms
    with FileLock("./data.lock"):
        train_dataset = torchvision.datasets.MNIST('./files/', train=True, download=True,
                    transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.1307,), (0.3081,))
                    ]))
    random_seed = config["seed"]
    L = [int(len(train_dataset) / n_workers)] * n_workers
    L[-1] += len(train_dataset) % n_workers
    partitions = random_split(train_dataset, L, generator=torch.Generator().manual_seed(random_seed))

    batch_test = config["test_batch"]
    if rank == 0:
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./files/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
            batch_size=batch_test, shuffle=False, **kwargs)
    else:
        test_loader = None

    return coder.partition(partitions, rank), test_loader

def dataset_creator_CIFAR(rank, coder, use_cuda, config):
    n_workers = coder.n
    s_workers = coder.s
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    from filelock import FileLock
    import torchvision.transforms as tt
    from torchvision.datasets import cifar

    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats,inplace=True)])
    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

    with FileLock("./data.lock"):
        train_dataset=cifar.CIFAR10('./cifar10',train=True,transform=train_tfms,download=True)
    random_seed = config["seed"]
    L = [int(len(train_dataset) / n_workers)] * n_workers
    L[-1] += len(train_dataset) % n_workers
    partitions = random_split(train_dataset, L, generator=torch.Generator().manual_seed(random_seed))

    batch_test = config["test_batch"]
    if rank == 0:
        test_dataset=cifar.CIFAR10('./cifar10',train=False,transform=valid_tfms)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_test, shuffle=False)
    else:
        test_loader = None

    return coder.partition(partitions, rank), test_loader

def dataset_creator_Fashion_MNIST(rank, coder, use_cuda, config):
    n_workers = coder.n
    s_workers = coder.s
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    from filelock import FileLock
    import torchvision.transforms as tt
    from torchvision.datasets import FashionMNIST

    with FileLock("./data.lock"):
        train_dataset = FashionMNIST(root='./FashionMNIST', train=True, download=True, transform=tt.Compose([tt.Grayscale(3), tt.ToTensor(),
 tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    random_seed = config["seed"]
    L = [int(len(train_dataset) / n_workers)] * n_workers
    partitions = random_split(train_dataset, L, generator=torch.Generator().manual_seed(random_seed))

    batch_test = config["test_batch"]
    if rank == 0:
        test_dataset = FashionMNIST(root='./FashionMNIST', train=False, download=True, transform=tt.Compose([tt.Grayscale(3), tt.ToTensor(),
 tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_test, shuffle=False)
    else:
        test_loader = None

    return coder.partition(partitions, rank), test_loader

def dataset_creator(rank, coder, use_cuda, config):
    if "dataset" in config:
        if config["dataset"] == "MNIST":
            return dataset_creator_MNIST(rank, coder, use_cuda, config)
        if config["dataset"] == "Fashion_MNIST":
            return dataset_creator_Fashion_MNIST(rank, coder, use_cuda, config)
        if config["dataset"] == "CIFAR":
            return dataset_creator_CIFAR(rank, coder, use_cuda, config)
    else:
        return dataset_creator_MNIST(rank, coder, use_cuda, config)

import torch.optim as optim

def build_coder(config):
    if config["coder"] == "FR":
        coder = GC_FR(config)
    if config["coder"] == "CR":
        coder = GC_CR(config)
    if config["coder"] == "IS-FR" or config["coder"] == "IS_FR":
        coder = IS_GC_FR(config)
    if config["coder"] == "IS-CR" or config["coder"] == "IS_CR":
        coder = IS_GC_CR(config)
    if config["coder"] == "IS-HR" or config["coder"] == "IS_HR":
        coder = IS_GC_HR(config)
    return coder

def build_model(config):
    if "model" in config:
        if config["model"] == "MLP":
            model = MLP()
        elif config["model"] == "ResNet" and config["dataset"] == "MNIST":
            model = ResNet_MNIST()
        elif config["model"] == "ResNet" and config["dataset"] == "CIFAR":
            model = ResNet_CIFAR()
        elif config["model"] == "ResNet" and config["dataset"] == "Fashion_MNIST":
            model = ResNet_Fashion_MNIST()
        else:
            model = CNN()
    else:
        model = CNN()
    return model

def build_model_name(config):
    if "model" in config:
        if config["model"] == "MLP":
            model = "mnist_mlp.pt"
        elif (config["model"] == "ResNet") and (config["dataset"] == "MNIST"):
            model = "mnist_resnet.pt"
        elif config["model"] == "ResNet" and config["dataset"] == "CIFAR":
            model = "cifar_resnet.pt"
        elif config["model"] == "ResNet" and config["dataset"] == "Fashion_MNIST":
            model = "fashion_mnist_resnet.pt"
        else:
            model = "mnist_cnn.pt"
    else:
        model = "mnist_cnn.pt"
    return model

def build_optimizer(config, model):
    if "optimizer" in config:
        if config["optimizer"] == "SGD":
            opt = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
        elif config["optimizer"] == "Adam":
            opt = optim.Adam(model[-1].parameters(), lr=config["lr"])
    else:
        opt = optim.SGD(model[-1].parameters(), lr=config["lr"], momentum=config["momentum"])
    return opt

class Network(object):
    def initialize(self, config, rank):
        self.config = config
        torch.set_default_dtype(torch.float64)
        seed_torch(self.config["seed"])
        self.rank = rank
        self.rng = np.random.default_rng(self.config["seed"] + rank)
        self.n_workers = self.config["n"]
        self.s_workers = self.config["s"]
        self.shuffle = list(range(0, self.n_workers))
        self.coder = build_coder(self.config)
        self.batch_train = self.config["train_batch"]
        self.batch_idx = 0
        self.train_datasets, self.test_loader = dataset_creator(self.rank, self.coder, self.use_cuda, self.config)
        kwargs = {"num_workers": 1, "pin_memory": True} if self.use_cuda else {}
        self.train_loaders = [torch.utils.data.DataLoader(self.train_datasets[i], batch_size=int(self.batch_train / self.n_workers), shuffle=True, generator=torch.Generator().manual_seed(self.config["seed"]), **kwargs) for i in range(len(self.train_datasets))]

        self.models = []
        self.optimizers = []
        for i in range(len(self.train_loaders)):
            model = build_model(config)
            self.models.append(model.to(self.device))
            self.optimizers.append(optim.SGD(self.models[-1].parameters(), lr=self.config["lr"], momentum=self.config["momentum"]))
        self.loss_correct = [None for i in range(len(self.train_loaders))]

    def update_batch_idx(self):
        self.batch_idx = (self.batch_idx + 1) % self.get_batches()
        if self.batch_idx == 0:
            kwargs = {"num_workers": 1, "pin_memory": True} if self.use_cuda else {}
            self.train_loaders = [torch.utils.data.DataLoader(dataset, batch_size=int(self.batch_train / self.n_workers), shuffle=True, generator=torch.Generator().manual_seed(self.config["seed"]+self.batch_idx), **kwargs) for dataset in self.train_datasets]

    def train(self):
        for i in range(len(self.train_loaders)):
            loss, corr = train_(self.batch_idx, self.models[i], self.device, self.train_loaders[i], self.optimizers[i], (self.rank, i))
            loss = loss.cpu()
            self.loss_correct[i] = np.array([loss.detach().numpy(), corr])
        self.update_batch_idx()

    def test(self):
        return test(self.models[0], self.device, self.test_loader)

    def get_gradients(self):
        import time
        if "s_workers" in self.config and "s_delay" in self.config and self.shuffle[self.rank] < self.config["s_workers"]:
            random.shuffle(self.shuffle)
            delay = self.rng.exponential(self.config["s_delay"])
            time.sleep(delay)

        return {'rank' : self.rank,
                'grad' : [self.coder.encode([self.optimizers[j].param_groups[0]['params'][i].grad.to(torch.device("cpu")) for j in range(len(self.optimizers))], self.rank) for i in range(len(self.optimizers[0].param_groups[0]['params']))],
                'loss_correct': self.coder.encode(self.loss_correct, self.rank)
        }

    def set_gradients(self, grads):
        for i in range(len(grads)):
            for j in range(len(self.optimizers)):
                self.optimizers[j].param_groups[0]['params'][i].grad = grads[i].type(torch.DoubleTensor)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def save(self):
        name = build_model_name(self.config)
        torch.save(self.models[0].state_dict(), config["path"] + "/" + name)
        print("saved")

    def load(self):
        name = build_model_name(self.config)
        for model in self.models:
            model.load_state_dict(torch.load(config["path"] + "/" + name))

    def get_batches(self):
        return len(self.train_loaders[0])

    def get_samples(self):
        return len(self.train_datasets[0])

    def pre_train(self):
        pass

@ray.remote(num_cpus=1)
class RemoteNetwork(Network):
    def __init__(self, config, rank):
        seed_torch(config["seed"])
        self.use_cuda = False
        self.device = device = torch.device("cpu")
        self.initialize(config, rank)

@ray.remote(num_gpus=1)
class GPURemoteNetwork(Network):
    def __init__(self, config, rank):
        seed_torch_gpu(config["seed"])
        self.use_cuda = True
        self.device = device = torch.device("cuda")
        self.initialize(config, rank)

    def set_gradients(self, grads):
        for i in range(len(grads)):
            for j in range(len(self.optimizers)):
                self.optimizers[j].param_groups[0]['params'][i].grad = grads[i].type(torch.DoubleTensor).cuda()

def train(config):
    import time
    timer = 0
    last_time = time.time()
    seed_torch(config["seed"])
    coder = build_coder(config)
    n_epochs = config["epoch"]
    c = 0
    useful_workers = 0
    loss = []
    corr = []
    repeat = 0
    for i in range(n_epochs):
        for j in range(int(batches)):
            [actor.train.remote() for actor in NetworkActors]
            grads = [actor.get_gradients.remote() for actor in NetworkActors]
            ready, remaining = ray.wait(grads, num_returns = n - s)
            grads = [ray.get(ref) for ref in ready]
            ranks = [grad['rank'] for grad in grads]
            loss_correct = [grad['loss_correct'] for grad in grads]
            grads = [grad['grad'] for grad in grads]

            from collections import OrderedDict
            start = np.random.randint(len(ranks))
            decoded_gradients = [coder.decode([grads[i][k] for i in range(len(grads))], ranks, start) for k in range(len(grads[0]))]
            uw = decoded_gradients[0][-1]
            decoded_loss_correct = coder.decode(loss_correct, ranks, start)
            averaged_gradients = [(g).type(torch.FloatTensor) for g, w in decoded_gradients]
            useful_workers += uw
            grads_id = ray.put(averaged_gradients)
            [actor.set_gradients.remote(grads_id) for actor in NetworkActors]
            [actor.step.remote() for actor in NetworkActors]
            c += 1
            curr_time = time.time()
            timer += curr_time - last_time
            loss.append(decoded_loss_correct[0][0] / uw)
            corr.append(decoded_loss_correct[0][1] / uw / (config["train_batch"] / config["n"]))
            if j % config["log_interval"] == 0:
                print('{:.0f}\t{:.0f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(i, i * batches + j, timer, np.mean(loss), np.mean(corr), useful_workers / (i * batches + j + 1)))
                if "step" in config and (i* batches + j >= config["step"]):
                    return
                if "accuracy" in config and "repeat" in config:
                    if np.mean(corr) >= config["accuracy"]:
                        repeat += 1
                    else:
                        repeat = 0
                    if "repeat" in config and repeat >= config["repeat"]:
                        return
                loss = []
                corr = []
            last_time = time.time()
        last_time = time.time()

import argparse

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

if __name__ == '__main__':
    config = {
        "mode": "cluster",
        "device": "cpu",
        "path": "/home/ubuntu/code",
        "log_interval": 10,
        "s_delay": 0,
        "s_workers": 0,
        "coder": "IS-FR",
        "n": 8,
        "s": 0,
        "c": 2,
        "c1": 0,
        "g": 3,
        "train_batch": 192,
        "test_batch": 1000,
        "lr": 0.01,
        "momentum": 0,
        "epoch": 5,
        "seed": 2
        # "correct": True,
        # "pipeline": True,
        # "step": 1
    }
    config_MLP_MNIST = {
        "accuracy": 0.9,
        "repeat": 3,
        "optimizer": "SGD",
        "model": "MLP",
        "dataset": "MNIST",
        "log_interval": 5,
        "train_batch": 64,
        "epoch": 10,
        "lr": 0.01,
        "momentum": 0
    }
    config_ResNet_MNIST = {
        "accuracy": 0.8,
        "repeat": 3,
        "optimizer": "SGD",
        "model": "ResNet",
        "dataset": "MNIST",
        "log_interval": 5,
        "train_batch": 64,
        "epoch": 10,
        "lr": 0.005,
        "momentum": 0
    }
    config_ResNet_CIFAR = {
        "optimizer": "SGD",
        "model": "ResNet",
        "dataset": "CIFAR",
        "epoch": 100,
        "train_batch": 128,
        "lr": 0.006,
        "accuracy": 0.7,
        "repeat": 1,
        "log_interval": 5
    }
    config_ResNet_Fashion_MNIST = {
        "optimizer": "SGD",
        "model": "ResNet",
        "dataset": "Fashion_MNIST",
        "epoch": 2,
        "lr": 0.01,
        "train_batch": 128,
        "accuracy": 0.9,
        "repeat": 3,
        "log_interval": 5
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    parser.add_argument('-m', '--mwargs', nargs='*', action=ParseKwargs)
    parser.add_argument('-r', '--rwargs', nargs='*', action=ParseKwargs)
    parser.add_argument('-c', '--cwargs', nargs='*', action=ParseKwargs)
    parser.add_argument('-f', '--fwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()
    if args.mwargs is not None:
        config.update(config_MLP_MNIST)
    if args.rwargs is not None:
        config.update(config_ResNet_MNIST)
    if args.cwargs is not None:
        config.update(config_ResNet_CIFAR)
    if args.fwargs is not None:
        config.update(config_ResNet_Fashion_MNIST)
    if args.kwargs is not None:
        for k, v in args.kwargs.items():
            if k == "coder" or k == "device" or k == "mode" or k == "path":
                config[k] = v
            elif k == "lr" or k == "momentum" or k == "accuracy" or k == "s_delay":
                config[k] = float(v)
            else:
                config[k] = int(v)
    print(config)

    n = config["n"]
    s = config["s"]

    if "mode" in config and config["mode"] == "cluster":
        ray.init(address="auto")
    else:
        ray.init(num_cpus=n)
    if "device" in config and (config["device"] == "gpu" or config["device"] == "cuda"):
        NetworkActors = [GPURemoteNetwork.remote(config, i) for i in range(n)]
    else:
        NetworkActors = [RemoteNetwork.remote(config, i) for i in range(n)]
    if "save" in config and config["save"] == 1:
        NetworkActors[0].save.remote()
        import time
        time.sleep(10)
        exit()
    for i in range(0, len(NetworkActors)):
        NetworkActors[i].load.remote()
    batches = ray.get(NetworkActors[0].get_batches.remote())
    samples = ray.get(NetworkActors[0].get_samples.remote())
    for actor in NetworkActors:
        actor.pre_train.remote()

    train(config)
