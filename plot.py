from matplotlib import pyplot as plt
import pickle

if __name__ == '__main__':
    path = r"C:\Users\11818\Desktop\RL\Code\PG4LQR\remote\logs\2-2\natural is True smoothing parameter 0.05 lr 0.001\run1\summary.pkl"
    data = pickle.load(open(path, mode="rb"))
    cost_ratio_ = [item[4] for item in data]
    cost_ratio = [sum(cost_ratio_[i: i + 50]) / 50 for i in range(len(cost_ratio_) - 50)]
    x_ = list(range(len(cost_ratio)))
    x = [item * 50 for item in x_]
    plt.plot(x, cost_ratio)
    plt.xlabel("Iterations")
    plt.ylabel(r"$\frac{C(K)-C(K^{*})}{C(K^{*})}$")
    plt.yscale('log')
    plt.grid()
    plt.savefig("result1.pdf", bbox_inches='tight')
    plt.close()