from matplotlib import pyplot as plt
import pickle

if __name__ == '__main__':
    path = r"C:\Users\11818\Desktop\RL\Code\PG4LQR\logs\2-2\natural is False smoothing parameter 0.05 lr 0.001\run1\summary.pkl"
    # path = r"C:\Users\11818\Desktop\RL\Code\PG4LQR\remote\logs\100-20\natural is False smoothing parameter 0.005 lr 0.0001\run2\summary.pkl"
    data = pickle.load(open(path, mode="rb"))
    cost_ratio_ = [item[4] for item in data][0: 1000]
    cost_ratio = [sum(cost_ratio_[i: i + 50]) / 50 for i in range(len(cost_ratio_) - 50)]
    x_ = list(range(len(cost_ratio)))
    x = [item * 50 for item in x_]
    # plt.plot(x, cost_ratio, label="gradient descent")

    path = r"C:\Users\11818\Desktop\RL\Code\PG4LQR\remote\logs\2-2\natural is True smoothing parameter 0.05 lr 0.001\run1\summary.pkl"
    # path = r"C:\Users\11818\Desktop\RL\Code\PG4LQR\remote\logs\100-20\natural is False smoothing parameter 0.005 lr 0.0001\run2\summary.pkl"
    data = pickle.load(open(path, mode="rb"))
    cost_ratio_ = [item[4] for item in data][0: 1000]
    cost_ratio1 = [sum(cost_ratio_[i: i + 50]) / 50 for i in range(len(cost_ratio_) - 50)]
    x_ = list(range(len(cost_ratio)))
    x1 = [item * 50 for item in x_]
    # plt.plot(x, cost_ratio, label="natural policy descent")

    plt.plot([cost_ratio[i]/cost_ratio1[i] for i in range(len(cost_ratio))])
    plt.xlabel("Iterations")
    plt.ylabel(r"$\frac{C(K)-C(K^{*})}{C(K^{*})}$")
    plt.title("state dim:2 action dim:2")
    # plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig("result10.pdf", bbox_inches='tight')
    plt.close()
