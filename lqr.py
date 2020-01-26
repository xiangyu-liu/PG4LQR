import numpy as np
from dynamics import Dynamics
import argparse


# np.random.seed(0)

def modify_dynamics4test(env):
    env.A = np.diag([1, ] * env.state_dim)
    env.B = np.diag([1, ] * env.action_dim)
    env.Q = 0.01 * np.diag([1, ] * env.state_dim)
    env.R = np.diag([0, ] * env.action_dim)


def main(args):
    env = Dynamics(args.state_dim, args.action_dim)
    # modify_dynamics4test(env)
    optimal_K = env.cal_optimal_K()
    K = np.zeros((args.action_dim, args.state_dim))
    d = args.state_dim

    for epoch in range(args.epoch):
        c_gradient = np.zeros(K.shape)
        sigma_gradient = np.zeros((args.state_dim, args.state_dim))
        for i in range(args.m):
            U_i = 2 * (np.random.rand(*K.shape) - 0.5)
            U_i = (args.r / np.linalg.norm(U_i)) * U_i
            state = env.reset()
            cost_list, state_list = env.rollout(K + U_i, state, args.l)
            C_i = sum(cost_list)
            sigma_i = sum([np.dot(i, i.T) for i in state_list])
            c_gradient += C_i * U_i
            sigma_gradient += sigma_i

        c_gradient *= (d / (args.m * args.r ** 2))
        sigma_gradient *= 1 / args.m
        gradient = np.dot(c_gradient, np.linalg.inv(sigma_gradient))
        # clip gradient to stablize training
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
        # print(sum(cost_list1), sum(cost_list2), (sum(cost_list1) - sum(cost_list2)) / sum(cost_list2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dim", default=1, type=int)
    parser.add_argument("--action_dim", default=1, type=int)
    parser.add_argument("--l", default=10, type=int, help="roll-out length")
    parser.add_argument("--m", default=100, type=int, help="number of trajectories")
    parser.add_argument("--r", default=0.005, type=float, help="smoothing parameter")
    parser.add_argument("--epoch", default=10000, type=int, help="number of training epochs")
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    args = parser.parse_args()
    main(args=args)
