import numpy as np
import matplotlib.pyplot as plt


def make_plot(xs, ys, color, label, xlabel, ylabel):
    plt.plot(xs, ys, color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')


nu = 1
tau = 1
tau_s = 1
u_spike = 1


def eps(s, d_ij):
    return (s - d_ij) / tau_s * np.exp(-(s - d_ij) / tau_s) * (1 if s - d_ij > 0 else 0)


def eta(s):
    return -nu * np.exp(-s / tau) * (1 if s > 0 else 0)


class SNN:
    def __init__(self, input_sz, hidden_sz, output_sz):
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz

        self.synapses_cnt1 = np.random.randint(low=1, high=6, size=(input_sz, hidden_sz))
        self.synapses_cnt2 = np.random.randint(low=1, high=6, size=(hidden_sz, output_sz))

        self.d1 = np.random.randint(low=1, high=8, size=(input_sz, hidden_sz))
        self.d2 = np.random.randint(low=1, high=8, size=(hidden_sz, output_sz))

        self.w1 = np.random.random((input_sz, hidden_sz))
        self.w2 = np.random.random((hidden_sz, output_sz))

        self.u1 = np.zeros((1000, hidden_sz))
        self.u2 = np.zeros((1000, output_sz))

        self.prev_t_0 = np.zeros(input_sz)
        self.prev_t_1 = np.zeros(hidden_sz)
        self.prev_t_2 = np.zeros(output_sz)

    def apply(self, input_layer, t):
        for i in range(input_sz):
            if input_layer[i] == 1:
                self.prev_t_0[i] = t

        for i in range(hidden_sz):
            self.u1[t, i] = eta(t - self.prev_t_1[i])
            for j in range(input_sz):
                self.u1[t, i] += self.w1[j, i] * eps(t - self.prev_t_0[j] - self.d1[j, i], self.d1[j, i])
            if self.u1[t, i] > u_spike:
                self.prev_t_1[i] = t

        for i in range(output_sz):
            self.u2[i] = eta(t - self.prev_t_2[i])
            for j in range(hidden_sz):
                self.u2[t, i] += self.w2[j, i] * eps(t - self.prev_t_1[j] - self.d2[j, i], self.d2[j, i])
            if self.u2[t, i] > u_spike:
                self.prev_t_2[i] = t

    def visualize_u(self, maxT):
        tim = np.arange(maxT)
        for i in range(hidden_sz):
            make_plot(tim, self.u1[:maxT, i], 'g', 'u(t)', 'time', 'u')
            plt.savefig(path + "/u_hidden_" + str(i) + "(t)")
            plt.close()

        for i in range(output_sz):
            make_plot(tim, self.u2[:maxT, i], 'g', 'u(t)', 'time', 'u')
            plt.savefig(path + "/u_out_" + str(i) + "(t)")
            plt.close()


if __name__ == "__main__":
    path = "data"
    input_sz = 20
    output_sz = 2

    hidden_sz = 10

    T = 100
    input_pattern = np.zeros((T * 10, input_sz))
    for i in range(input_sz):
        j = -1
        while j < 100:
            lam = 0.05
            j += int(-np.log(1. - np.random.rand()) / lam) + 1
            if j >= 100:
                break
            input_pattern[j, i] = 1

    # print(input_pattern)

    snn1 = SNN(input_sz, hidden_sz, output_sz)

    epoch_cnt = 5
    for epoch in range(epoch_cnt):
        for tt in range(T):
            snn1.apply(input_pattern[tt], epoch * T + tt)

    snn1.visualize_u(T * epoch_cnt)
