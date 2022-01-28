# On Arbitrary Ignorance of Stragglers with Gradient Coding
This repository contains source code for reproduce the experimental result in the paper submission of "On Arbitrary Ignorance of Stragglers with Gradient Coding".

## Usage

```
python3 ray_gc.py -r mode={MODE} device={DEVICE} path={PATH} log_internal={LOG_INTERVAL} s_delay={S_DELAY} s_workers={S_WORKERS} coder={CODER} n={N} s={S} c={C} c1={C1} g={G} train_batch={TRAIN_BATCH} lr={LR} seed={SEED}
```

- MODE: local or cluster
- DEVICE: cpu or gpu

- PATH: the path for loading/saving model parameters

- LOG_INTERVAL: the interval of steps in the output

- S_DELAY: \lambda in the exponential distribution of additional stragglers

- S_WORKERS: the number of additional stragglers

- CODER: FR or CR or IS-FR or IS-CR or IS-HR

- N: # workers

- S: # stragglers to tolerate

- C: storage overhead

- C1: c1 in IS-HR

- g: g in IS-HR

- TRAIN_BATCH: batch size in training

- LR: learning rate

- SEED: seed or random generators

Use -k, -m, -r, -c to pre-load parameters of MLP + MNIST, ResNet-18 + MNIST, ResNet-18 + CIFAR10, ResNet-18 + FashionMNIST.

