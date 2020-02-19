import numpy as np
from dynamics import Dynamics
import argparse
from pathlib import Path
import os
from tensorboardX import SummaryWriter
import pickle

np.random.seed(0)


class Adam:
    '''
    implementation of adam optimizer
    '''

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        self.m += (1 - self.beta1) * (grads - self.m)
        self.v += (1 - self.beta2) * (grads ** 2 - self.v)
        params -= lr_t * self.m / (np.sqrt(self.v) + 1e-7)
        return params


# this function is only for test usage
def modify_dynamics4test(env):
    env.A = np.diag([1, ] * env.state_dim)
    env.B = np.diag([1, ] * env.action_dim)
    env.Q = 0.01 * np.diag([1, ] * env.state_dim)
    env.R = np.diag([0, ] * env.action_dim)


def main(args):
    model_dir = Path('./logs') / "{}-{}".format(args.state_dim,
                                                args.action_dim) / "natural is {} smoothing parameter {} lr {}".format(
        args.natural, args.r,
        args.lr)
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    os.makedirs(str(run_dir))
    logger = SummaryWriter(str(run_dir))
    print("making dir {}".format((str(run_dir))))

    adam = Adam(lr=args.lr)
    env = Dynamics(args.state_dim, args.action_dim)
    # calculate the optimal policy
    optimal_K = env.cal_optimal_K()
    K = np.zeros((args.action_dim, args.state_dim))
    d = args.state_dim

    result_list = []
    for epoch in range(args.epoch):
        c_gradient = np.zeros(K.shape)
        sigma_gradient = np.zeros((args.state_dim, args.state_dim))
        for i in range(args.m):
            # give the policy a random perturbation
            U_i = 2 * (np.random.rand(*K.shape) - 0.5)
            U_i = (args.r / np.linalg.norm(U_i)) * U_i
            state = env.reset()
            # roll out the policy with perturbation
            cost_list, state_list = env.rollout(K + U_i, state, args.l)
            C_i = sum(cost_list)
            sigma_i = sum([np.dot(i, i.T) for i in state_list])
            # estimate the gradients and covariance matrix
            c_gradient += C_i * U_i
            old_sigma = sigma_gradient
            sigma_gradient += sigma_i
            # sigma_norm_diff = np.linalg.norm(
            #     np.linalg.inv(old_sigma / i) - np.linalg.inv(sigma_gradient / (i + 1))) / np.linalg.norm(
            #     np.linalg.inv(old_sigma / i))
        #     if not (i==0):
        #         print("{}th iteration sigma norm diff is {}".format(i, sigma_norm_diff))
        # print('\n')
        c_gradient *= (d / (args.m * args.r ** 2))
        sigma_gradient *= 1 / args.m
        if not args.natural:
            gradient = c_gradient
        else:
            gradient = np.dot(c_gradient, np.linalg.inv(sigma_gradient))

        # clip gradient to stablize training (optional)
        if np.linalg.norm(gradient) >= 10:
            gradient *= (10 / np.linalg.norm(gradient))

        # the usage of adam optimizer is optional
        # if you do not want this you can just un-comment the next line and comment the next of the next one
        # K = K - args.lr * gradient
        K = adam.update(K, gradient)

        if epoch % 50 == 0:
            cost_stat1 = []
            cost_stat2 = []
            # evaluate the current policy
            for tmp in range(10):
                state = env.reset()
                cost_list1, _ = env.rollout(K, state, 20, True)
                cost_list2, _ = env.rollout(optimal_K, state, 20, True)
                cost_stat1.append(sum(cost_list1))
                cost_stat2.append(sum(cost_list2))
            cost_list1 = cost_stat1
            cost_list2 = cost_stat2
            print(
                "epoch is {}\ngradient norm is {}\nK norm is {}\noptimal K norm is {}\nnorm of difference is {}\ncost difference ratio is {}\n".format(
                    epoch,
                    np.linalg.norm(gradient),
                    np.linalg.norm(K),
                    np.linalg.norm(optimal_K),
                    np.linalg.norm(K - optimal_K) / np.linalg.norm(optimal_K),
                    ((sum(cost_list1) - sum(cost_list2)) / sum(cost_list2))[0, 0]))
            logger.add_scalar("gradient norm", np.linalg.norm(gradient), epoch)
            logger.add_scalar("K norm", np.linalg.norm(K), epoch)
            logger.add_scalar("optimal K norm", np.linalg.norm(optimal_K), epoch)
            logger.add_scalar("norm of difference", np.linalg.norm(K - optimal_K) / np.linalg.norm(optimal_K), epoch)
            logger.add_scalar("cost difference", ((sum(cost_list1) - sum(cost_list2)) / sum(cost_list2))[0, 0], epoch)
            result_list.append([np.linalg.norm(gradient), np.linalg.norm(K), np.linalg.norm(optimal_K),
                                np.linalg.norm(K - optimal_K) / np.linalg.norm(optimal_K),
                                ((sum(cost_list1) - sum(cost_list2)) / sum(cost_list2))[0, 0]])
        if epoch % 1000 == 0:
            pickle.dump(result_list, open(run_dir / "summary.pkl", mode="wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dim", default=100, type=int)
    parser.add_argument("--action_dim", default=20, type=int)
    parser.add_argument("--l", default=30, type=int, help="roll-out length")
    parser.add_argument("--m", default=300, type=int, help="number of trajectories")
    parser.add_argument("--r", default=0.005, type=float, help="smoothing parameter")
    parser.add_argument("--epoch", default=1000000, type=int, help="number of training epochs")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--natural", default=False, action="store_true")
    args = parser.parse_args()
    main(args=args)
