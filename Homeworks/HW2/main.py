import numpy as np
import matplotlib.pyplot as plt
import random


def make_plot(xs, ys, color, label, xlabel, ylabel):
    plt.plot(xs, ys, color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')


def will_activate(prob):
    return random.random() < prob


if __name__ == "__main__":
    s0 = 1000
    s_max = 100000
    v_deg = 100
    v_uptake = 1000
    u0 = -70
    u_spike = 30
    u_r2 = 0.05
    u_r3 = 0.1
    u_r4 = 0.13
    p_r2 = 0.6
    p_r3 = 0.71
    p_r4 = 0.97
    T = 10000
    r = 1500

    path = "./data"

    v_rel = 1101
    x_r2 = 500
    x_r3 = 500
    x_r4 = r - x_r2 - x_r3

    time_quant_ms = 5
    N = T // time_quant_ms
    tim = np.zeros(N)
    s = np.zeros(N)
    p1_r2 = np.zeros(N)
    p1_r3 = np.zeros(N)
    p1_r4 = np.zeros(N)
    q_r2 = np.zeros(N)
    q_r3 = np.zeros(N)
    q_r4 = np.zeros(N)
    u = np.zeros(N)
    f_activation = np.zeros(N)

    s[0] = s0
    u[0] = u0
    spikes_cnt = 0

    for i in range(1, N):
        t = i * time_quant_ms
        tim[i] = t
        s[i] = s0 + (v_rel - v_deg - v_uptake) * t

        p1_r2[i] = s[i] * p_r2 / s_max
        p1_r3[i] = s[i] * p_r3 / s_max
        p1_r4[i] = s[i] * p_r4 / s_max

        q_r2[i] = 0
        for j in range(x_r2):
            q_r2[i] += will_activate(p1_r2[i])
        q_r3[i] = 0
        for j in range(x_r3):
            q_r3[i] += will_activate(p1_r3[i])
        q_r4[i] = 0
        for j in range(x_r4):
            q_r4[i] += will_activate(p1_r4[i])

        u[i] = u[i - 1] + (u_r2 * q_r2[i] + u_r3 * q_r3[i] + u_r4 * q_r4[i]) * time_quant_ms
        if u[i] >= u_spike:
            f_activation[i] = 1
            u[i] = u0
            spikes_cnt += 1
        else:
            f_activation[i] = 0

    print(spikes_cnt)

    make_plot(tim, s, 'g', 's(t)', 'time', 's')
    plt.savefig(path + "/s(t)")
    plt.close()

    make_plot(tim, u, 'g', 'u(t)', 'time', 'u')
    plt.savefig(path + "/u(t)")
    plt.close()

    make_plot(tim, q_r2, 'g', 'q_r2(t)', 'time', 'q_r2')
    make_plot(tim, q_r3, 'b', 'q_r3(t)', 'time', 'q_r3')
    make_plot(tim, q_r4, 'y', 'q_r4(t)', 'time', 'q_r4')
    plt.savefig(path + "/q(t)")
    plt.close()
