# PG4LQR
This is an implementation of model-free policy gradient algorithm for LQR

The structure of this project is shown as follows

```
PG4LQR
|--dynamics.py: implementation of LQR system
|--lqr.py: implementation of model-free algorithm plus the adam optimizer (optional)
|--parallel_lqr.py: the multi-process version (the algorithm is the same as lqr.py)
|--plot.py: plot figures according to saved results
```

## Usage

```shell
python lqr.py --action_dim xx --state_dim xx --lr xx --epoch xx --r xx --natural
```

- action_dim: dimension of inputs
- state_dim: dimensional of states
- lr: step size, also known as the learning rate
- epoch: total number of training iterations
- r: the smoothing parameter
- natural: adding this argument means using natural policy gradient (gradient descent as default)

Currently, the single-process version is more efficient. You can just ignore parallel version