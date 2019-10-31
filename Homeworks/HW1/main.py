import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def a_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-0.1 * (V + 40)))


def a_h(V):
    return 0.07 * np.exp(-0.05 * (V + 65))


def b_m(V):
    return 4 * np.exp(-0.0556 * (V + 65))


def b_h(V):
    return 1 / (1 + np.exp(-0.1 * (V + 35)))


def a_n(V):
    return 0.01 * (V + 55) / (1 - np.exp(-0.1 * (V + 55)))


def b_n(V):
    return 0.125 * np.exp(-0.0125 * (V + 65))


t1 = 30
t2 = 50
t3 = 90
t4 = 100
t_zero = 50


def calc_I_inj_1(N, T, t):
    res_ms = np.zeros(T + 1)

    I1 = 8 / 1000
    I2 = 15 / 1000
    I3 = 32 / 1000
    I4 = 47 / 1000

    res_ms[t_zero:t_zero + t1] = I1
    res_ms[t1 + 2 * t_zero:t1 + t2 + 2 * t_zero] = I2
    res_ms[t1 + t2 + 3 * t_zero:t1 + t2 + t3 + 3 * t_zero] = I3
    res_ms[t1 + t2 + t3 + 4 * t_zero:t1 + t2 + t3 + t4 + 4 * t_zero] = I4

    res = np.zeros(N)
    for i in range(N):
        res[i] = res_ms[int(t[i])]

    return res


def calc_I_inj_2(N, T, t):
    res_ms = np.zeros(T + 1)
    I = [0, 60 / 1000, 0, 180 / 1000]
    for i in range(T):
        res_ms[i] = I[i % 4]

    res = np.zeros(N)
    for i in range(N):
        res[i] = res_ms[int(t[i])]

    return res


def model(z, t, I):
    V = z[0]
    n = z[1]
    m = z[2]
    h = z[3]

    dVdt = (I - (g_K * n ** 4 * (V - E_K) + g_Na * m ** 3 * h * (V - E_Na) + g_L * (V - E_L))) / c_m
    dndt = a_n(V) * (1 - n) - b_n(V) * n
    dmdt = a_m(V) * (1 - m) - b_m(V) * m
    dhdt = a_h(V) * (1 - h) - b_h(V) * h
    dzdt = [dVdt, dndt, dmdt, dhdt]

    return dzdt


def solve(I_e, V, n, m, h, z0):
    V[0] = z0[0]
    n[0] = z0[1]
    m[0] = z0[2]
    h[0] = z0[3]
    for i in range(1, N):
        tspan = [t[i - 1], t[i]]
        z = odeint(model, z0, tspan, args=(I_e[i],))
        V[i] = z[1][0]
        n[i] = z[1][1]
        m[i] = z[1][2]
        h[i] = z[1][3]
        z0 = z[1]


def save_plots(path, I_e, V, n, m, h):
    plt.plot(t, I_e, 'g', label='I_e(t)')
    plt.xlabel('time')
    plt.ylabel('values')
    plt.legend(loc='best')
    plt.savefig(path + "/I_e(t)")
    plt.close()

    plt.plot(t, V, 'g', label='V(t)')
    plt.xlabel('time')
    plt.ylabel('values')
    plt.legend(loc='best')
    plt.savefig(path + "/V(t)")
    plt.close()

    plt.plot(t, n, 'b', label='n(t)')
    plt.xlabel('time')
    plt.ylabel('values')
    plt.legend(loc='best')

    plt.plot(t, m, 'r', label='m(t)')
    plt.xlabel('time')
    plt.ylabel('values')
    plt.legend(loc='best')

    plt.plot(t, h, 'y', label='h(t)')
    plt.xlabel('time')
    plt.ylabel('values')
    plt.legend(loc='best')
    plt.savefig(path + "/n(t), m(t), h(t)")
    plt.close()


def save_phase_space_plots(path, I_e, V, n, m, h):
    time_start = np.array([t_zero, 2 * t_zero + t1, t1 + t2 + 3 * t_zero, t1 + t2 + t3 + 4 * t_zero]) * N // (T + 1)
    time_end = np.array(
        [t_zero + t1, 2 * t_zero + t1 + t2, t1 + t2 + 3 * t_zero + t3, t1 + t2 + t3 + 4 * t_zero + t4]) * N // (T + 1)

    for i in range(len(time_start)):
        plt.plot(n[time_start[i]:time_end[i]], V[time_start[i]:time_end[i]], 'g', label='V(n)')
        plt.xlabel('time')
        plt.ylabel('values')
        plt.legend(loc='best')
        plt.savefig(path + "/V(n)_" + str(i))
        plt.close()

        plt.plot(m[time_start[i]:time_end[i]], V[time_start[i]:time_end[i]], 'g', label='V(m)')
        plt.xlabel('time')
        plt.ylabel('values')
        plt.legend(loc='best')
        plt.savefig(path + "/V(m)_" + str(i))
        plt.close()

        plt.plot(h[time_start[i]:time_end[i]], V[time_start[i]:time_end[i]], 'g', label='V(h)')
        plt.xlabel('time')
        plt.ylabel('values')
        plt.legend(loc='best')
        plt.savefig(path + "/V(h)_" + str(i))
        plt.close()


if __name__ == "__main__":
    g_L = 0.003
    g_K = 0.36
    g_Na = 1.2
    E_L = -54.387
    E_K = -77
    E_Na = 50
    c_m = 0.01

    z0 = [-65, 0.31, 0.053, 0.6]  # V0, n0, m0, h0

    N = 10000
    T = 500

    # time points
    t = np.linspace(0, T, N)
    I_e = calc_I_inj_1(N, T, t)

    V = np.empty_like(t)
    n = np.empty_like(t)
    m = np.empty_like(t)
    h = np.empty_like(t)

    solve(I_e, V, n, m, h, z0)
    path = "data/simulation1"
    save_plots(path, I_e, V, n, m, h)
    save_phase_space_plots(path, I_e, V, n, m, h)

    z0 = [-65, 0.3, 0.05, 0.6]  # V0, n0, m0, h0
    N = 10000
    T = 200
    t = np.linspace(0, T, N)
    I_e = calc_I_inj_2(N, T, t)

    V = np.empty_like(t)
    n = np.empty_like(t)
    m = np.empty_like(t)
    h = np.empty_like(t)

    solve(I_e, V, n, m, h, z0)
    path = "data/simulation2"
    save_plots(path, I_e, V, n, m, h)
