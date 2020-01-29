import numpy as np
np.random.seed(0)
import scipy
import scipy.linalg


# warnings.simplefilter('error', RuntimeWarning)


class Dynamics:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.initialize_dynamics()

    def initialize_dynamics(self):
        self.A = np.random.rand(*[self.state_dim, self.state_dim])
        self.B = np.random.rand(*[self.state_dim, self.action_dim])

        A_eig_max = max([i.real for i in np.linalg.eigvals(self.A) if i.imag == 0])
        # scale the matrix A such that the system can be more stable
        if A_eig_max >= 1:
            # self.A = self.A - (A_eig_max - 0.5) * np.diag([1, ] * self.state_dim)
            self.A = self.A / (A_eig_max)

        self.Q = np.random.rand(*[self.state_dim, self.state_dim])
        self.Q = (self.Q + self.Q.T) / 2
        Q_eig_max = max(i.real for i in np.linalg.eigvals(self.Q) if i.imag == 0)
        Q_eig_min = min(i.real for i in np.linalg.eigvals(self.Q) if i.imag == 0)
        # make sure Q is definite and the scale is not too large
        if Q_eig_min < 0:
            self.Q = self.Q - Q_eig_min * np.diag([1, ] * self.state_dim)
            self.Q = self.Q / (Q_eig_max - Q_eig_min)

        self.R = np.random.rand(*[self.action_dim, self.action_dim])
        self.R = (self.R + self.R.T) / 2
        R_eig_max = max(i.real for i in np.linalg.eigvals(self.R) if i.imag == 0)
        R_eig_min = min(i.real for i in np.linalg.eigvals(self.R) if i.imag == 0)
        # make sure R is definite and the scale is not too large
        if R_eig_min < 0:
            self.R = self.R - R_eig_min * np.diag([1, ] * self.action_dim)
            self.R = self.R / (R_eig_max - R_eig_min)

    def reset(self):
        self.state = np.random.rand(*[self.state_dim, 1])
        return self.state

    def step(self, action):
        reward = np.linalg.multi_dot([self.state.T, self.Q, self.state]) + np.linalg.multi_dot(
            [action.T, self.R, action])
        self.state = np.dot(self.A, self.state) + np.dot(self.B, action)

        return self.state, reward

    def cal_optimal_K(self):
        P = self.cal_approx_P()
        return -np.linalg.multi_dot(
            [np.linalg.inv(self.R + np.linalg.multi_dot([self.B.T, P, self.B])), self.B.T, P, self.A])

    def cal_approx_P(self):
        return scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)

    def cal_approx_P_manually(self):
        '''
        calculate P manually in a iterative approach
        :return: P
        '''
        P_old = self.Q
        i = 0
        while True:
            i += 1
            P_new = self.Q + np.linalg.multi_dot([self.A.T, P_old, self.A]) - np.linalg.multi_dot(
                [self.A.T, P_old, self.B, np.linalg.inv(self.R + np.linalg.multi_dot([self.B.T, P_old, self.B])),
                 self.B.T, P_old, self.A])

            ratio = np.linalg.norm(P_new - P_old) / np.linalg.norm(P_old)
            print(ratio, np.linalg.norm(P_new - P_old), np.linalg.norm(P_old))
            P_old = P_new
            if ratio <= 1e-5:
                print("iteration is:", i)
                break
        return P_new

    def rollout(self, K, state, l):
        '''
        :param K: the policy matrix
        :param state: the start state
        :param l: the roll-out length
        :return: cost sequence and state sequence
        '''
        self.state = state
        state_list = []
        cost_list = []
        for i in range(l):
            state_list.append(state)
            action = np.dot(K, state)
            state, cost = self.step(action)
            cost_list.append(cost)
        return cost_list, state_list
