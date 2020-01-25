import numpy as np
from dynamics import Dynamics
import argparse


# np.random.seed(0)

def main(args):
    env = Dynamics(args.state_dim, args.action_dim)
    optimal_K = env.cal_optimal_K()
    K = np.zeros((args.action_dim, args.state_dim))
    # K = optimal_K
    d = args.state_dim

    for epoch in range(args.epoch):
        c_gradient = np.zeros(K.shape)
        sigma_gradient = np.zeros((args.state_dim, args.state_dim))
        for i in range(args.m):
            U_i = np.random.rand(*K.shape)
            U_i = (args.r / np.linalg.norm(U_i)) * U_i

            C_i = 0
            sigma_i = np.zeros((args.state_dim, args.state_dim))
            state = env.reset()
            for j in range(args.l):
                action = -np.dot(K, state)
                state, cost = env.step(action)
                C_i += cost
                sigma_i += np.dot(state, state.T)
                # print("norm of actions and states", np.linalg.norm(action), np.linalg.norm(state))

            c_gradient += C_i * U_i
            sigma_gradient += sigma_i

        c_gradient *= (d / (args.m * args.r ** 2))
        sigma_gradient *= 1 / args.m
        gradient = np.dot(c_gradient, np.linalg.inv(sigma_gradient))
        # clip gradient to stablize training
        if np.linalg.norm(gradient) >= 10:
            gradient *= (10 / np.linalg.norm(gradient))
        K = K - args.lr * gradient
        print(epoch, np.linalg.norm(K), np.linalg.norm(optimal_K), np.linalg.norm(K - optimal_K) / np.linalg.norm(optimal_K))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dim", default=100, type=int)
    parser.add_argument("--action_dim", default=20, type=int)
    parser.add_argument("--l", default=100, type=int, help="roll-out length")
    parser.add_argument("--m", default=100, type=int, help="number of trajectories")
    parser.add_argument("--r", default=0.01, type=float, help="smoothing parameter")
    parser.add_argument("--epoch", default=10000, type=int, help="number of training epochs")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    args = parser.parse_args()
    main(args=args)
