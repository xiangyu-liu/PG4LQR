import numpy as np
from dynamics import Dynamics
import argparse
from pathlib import Path
import os
from tensorboardX import SummaryWriter
import multiprocessing
from multiprocessing import Pool

np.random.seed(0)
record1 = []
record2 = []
lock = multiprocessing.Lock()

def modify_dynamics4test(env):
    env.A = np.diag([1, ] * env.state_dim)
    env.B = np.diag([1, ] * env.action_dim)
    env.Q = 0.01 * np.diag([1, ] * env.state_dim)
    env.R = np.diag([0, ] * env.action_dim)


def subprocess(K, env):
    U_i = 2 * (np.random.rand(*K.shape) - 0.5)
    U_i = (args.r / np.linalg.norm(U_i)) * U_i
    state = env.reset()
    cost_list, state_list = env.rollout(K + U_i, state, args.l)
    C_i = sum(cost_list)
    sigma_i = sum([np.dot(i, i.T) for i in state_list])
    lock.acquire()
    record1.append(C_i * U_i)
    record2.append(sigma_i)
    lock.release()


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
    env = Dynamics(args.state_dim, args.action_dim)
    # modify_dynamics4test(env)
    optimal_K = env.cal_optimal_K()
    K = np.zeros((args.action_dim, args.state_dim))
    d = args.state_dim

    for epoch in range(args.epoch):
        record1.clear()
        record2.clear()
        p = Pool(args.l)
        for i in range(args.m):
            p.apply_async(subprocess, args=(K, env))
            # U_i = 2 * (np.random.rand(*K.shape) - 0.5)
            # U_i = (args.r / np.linalg.norm(U_i)) * U_i
            # state = env.reset()
            # cost_list, state_list = env.rollout(K + U_i, state, args.l)
            # C_i = sum(cost_list)
            # sigma_i = sum([np.dot(i, i.T) for i in state_list])
            # c_gradient += C_i * U_i
            # sigma_gradient += sigma_i
        p.close()
        p.join()
        c_gradient = sum(record1) * (d / (args.m * args.r ** 2))
        sigma_gradient = sum(record2) * (1 / args.m)
        if not args.natural:
            gradient = c_gradient
        else:
            gradient = np.dot(c_gradient, np.linalg.inv(sigma_gradient))

        # clip gradient to stablize training (optional)
        if np.linalg.norm(gradient) >= 10:
            gradient *= (10 / np.linalg.norm(gradient))

        K = K - args.lr * gradient
        state = env.reset()
        cost_list1, _ = env.rollout(K, state, 10)
        cost_list2, _ = env.rollout(optimal_K, state, 10)
        if epoch % 50 == 0:
            print(
                "epoch is {}\ngradient norm is {}\nK norm is {}\noptimal K norm is {}\nnorm of difference is {}\n".format(
                    epoch,
                    np.linalg.norm(gradient),
                    np.linalg.norm(K),
                    np.linalg.norm(
                        optimal_K),
                    np.linalg.norm(
                        K - optimal_K) / np.linalg.norm(
                        optimal_K)))
            logger.add_scalar("gradient norm", np.linalg.norm(gradient), epoch)
            logger.add_scalar("K norm", np.linalg.norm(K), epoch)
            logger.add_scalar("optimal K norm", np.linalg.norm(optimal_K), epoch)
            logger.add_scalar("norm of difference", np.linalg.norm(K - optimal_K) / np.linalg.norm(optimal_K), epoch)
        # print(sum(cost_list1), sum(cost_list2), (sum(cost_list1) - sum(cost_list2)) / sum(cost_list2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dim", default=2, type=int)
    parser.add_argument("--action_dim", default=2, type=int)
    parser.add_argument("--l", default=10, type=int, help="roll-out length")
    parser.add_argument("--m", default=100, type=int, help="number of trajectories")
    parser.add_argument("--r", default=0.05, type=float, help="smoothing parameter")
    parser.add_argument("--epoch", default=100000, type=int, help="number of training epochs")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--natural", default=True, action="store_true")
    args = parser.parse_args()
    main(args=args)
