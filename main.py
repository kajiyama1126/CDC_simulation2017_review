# -*- coding: utf-8 -*-
import matplotlib.pylab as plt
import numpy as np

from agent import Agent, Agent_moment_CDC2017, Agent_moment_CDC2017_s,Agent_L2,Agent_moment_CDC2017_L2,Agent_Dist,Agent_moment_CDC2017_Dist
from make_communication import Communication
from problem import Lasso_problem, Ridge_problem, Dist_problem


def L1_optimal_value(x_i, p, n, m, lamb):
    """
    :param x_i: float
    :param p:float
    :param n:int
    :param m:int
    :param lamb:float
    :return:float
    """
    p_all = np.reshape(p, (-1,))
    c = np.ones(n)
    d = np.reshape(c, (n, -1))
    A = np.kron(d, np.identity(m))
    tmp = np.dot(A, x_i) - p_all
    L1 = lamb * n * np.linalg.norm(x_i, 1)
    f_opt = 1 / 2 * (np.linalg.norm(tmp)) ** 2 + L1
    return f_opt


def L2_optimal_value(x_i, p, n, m, lamb):
    """
    :param x_i: float
    :param p:float
    :param n:int
    :param m:int
    :param lamb:float
    :return:float
    """
    p_all = np.reshape(p, (-1,))
    c = np.ones(n)
    d = np.reshape(c, (n, -1))
    A = np.kron(d, np.identity(m))
    tmp = np.dot(A, x_i) - p_all
    L2 = lamb * n * np.linalg.norm(x_i, 2) ** 2
    f_opt = 1 / 2 * (np.linalg.norm(tmp)) ** 2 + L2
    return f_opt


def Dist_optimal_value(x_i, p, n, m, lamb):
    """
    :param x_i: float
    :param p:float
    :param n:int
    :param m:int
    :param lamb:float
    :return:float
    """
    f_opt = 0
    for i in range(n):
        f_opt += np.linalg.norm(x_i-p[i])
    # p_all = np.reshape(p, (-1,))
    # c = np.ones(n)
    # d = np.reshape(c, (n, -1))
    # A = np.kron(d, np.identity(m))
    # tmp = np.dot(A, x_i) - p_all
    # f_opt = np.linalg.norm(tmp)
    return f_opt


def optimal_L1(n, m, lamb):
    p = [np.random.randn(m) for i in range(n)]
    # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
    p_num = np.array(p)
    # np.reshape(p)
    prob = Lasso_problem(n, m, p_num, lamb)
    prob.solve()
    x_opt = np.array(prob.x.value)  # 最適解
    x_opt = np.reshape(x_opt, (-1,))  # reshape
    f_opt = prob.send_f_opt()
    return p, x_opt, f_opt

def optimal_L2(n, m, lamb):
    p = [np.random.randn(m) for i in range(n)]
    # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
    p_num = np.array(p)
    # np.reshape(p)
    prob = Ridge_problem(n, m, p_num, lamb)
    prob.solve()
    x_opt = np.array(prob.x.value)  # 最適解
    x_opt = np.reshape(x_opt, (-1,))  # reshape
    f_opt = prob.send_f_opt()
    return p, x_opt, f_opt

def optimal_Dist(n, m, lamb):
    p = [np.random.randn(m) for i in range(n)]
    # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
    p_num = np.array(p)
    # np.reshape(p)
    prob = Dist_problem(n, m, p_num, lamb)
    prob.solve()
    x_opt = np.array(prob.x.value)  # 最適解
    x_opt = np.reshape(x_opt, (-1,))  # reshape
    f_opt = prob.send_f_opt()
    return p, x_opt, f_opt


def iteration_L1(n, m, p, lamb, test, P_history, f_opt, pattern):
    Agents = []
    for i in range(n):
        if pattern == 0:
            Agents.append(Agent(n, m, p[i], lamb, name=i, weight=None))
        elif pattern == 1:
            Agents.append(Agent_moment_CDC2017(n, m, p[i], lamb, name=i, weight=None))
        elif pattern == 2:
            Agents.append(Agent_moment_CDC2017_s(n, m, p[i], lamb, name=i, weight=None))

    f_error_history = []
    for k in range(test):
        # グラフの時間変化
        for i in range(n):
            Agents[i].weight = P_history[k][i]

        for i in range(n):
            for j in range(n):
                x_i, name = Agents[i].send()
                Agents[j].receive(x_i, name)

        for i in range(n):
            Agents[i].update(k)

        # x_ave = 0
        # for i in range(n):
        #     x_ave += 1.0/n * Agents[i].x_i
        f_value = []
        for i in range(n):
            x_i = Agents[i].x_i
            estimate_value = L1_optimal_value(x_i, p, n, m, lamb)
            f_value.append(estimate_value)

        # x_error_history[agent].append(np.linalg.norm(Agents[0].x_i- x_opt)**2)
        f_error_history.append(np.max(f_value) - f_opt)

    return f_error_history


def iteration_L2(n, m, p, lamb, test, P_history, f_opt, pattern):
    Agents = []
    for i in range(n):
        if pattern == 0:
            Agents.append(Agent_L2(n, m, p[i], lamb, name=i, weight=None))
        elif pattern == 1:
            Agents.append(Agent_moment_CDC2017_L2(n, m, p[i], lamb, name=i, weight=None))
        elif pattern == 2:
            Agents.append(Agent_moment_CDC2017_s(n, m, p[i], lamb, name=i, weight=None))

    f_error_history = []
    for k in range(test):
        # グラフの時間変化
        for i in range(n):
            Agents[i].weight = P_history[k][i]

        for i in range(n):
            for j in range(n):
                x_i, name = Agents[i].send()
                Agents[j].receive(x_i, name)

        for i in range(n):
            Agents[i].update(k)

        # x_ave = 0
        # for i in range(n):
        #     x_ave += 1.0/n * Agents[i].x_i
        f_value = []
        for i in range(n):
            x_i = Agents[i].x_i
            estimate_value = L2_optimal_value(x_i, p, n, m, lamb)
            f_value.append(estimate_value)

        # x_error_history[agent].append(np.linalg.norm(Agents[0].x_i- x_opt)**2)
        f_error_history.append(np.max(f_value) - f_opt)

    return f_error_history

def iteration_Dist(n, m, p, lamb, test, P_history, f_opt, pattern):
    Agents = []
    for i in range(n):
        if pattern == 0:
            Agents.append(Agent_Dist(n, m, p[i], lamb, name=i, weight=None))
        elif pattern == 1:
            Agents.append(Agent_moment_CDC2017_Dist(n, m, p[i], lamb, name=i, weight=None))
        elif pattern == 2:
            Agents.append(Agent_moment_CDC2017_s(n, m, p[i], lamb, name=i, weight=None))

    f_error_history = []
    for k in range(test):
        # グラフの時間変化
        for i in range(n):
            Agents[i].weight = P_history[k][i]

        for i in range(n):
            for j in range(n):
                x_i, name = Agents[i].send()
                Agents[j].receive(x_i, name)

        for i in range(n):
            Agents[i].update(k)

        f_value = []
        for i in range(n):
            x_i = Agents[i].x_i
            estimate_value = Dist_optimal_value(x_i, p, n, m, lamb)
            f_value.append(estimate_value)

        # x_error_history[agent].append(np.linalg.norm(Agents[0].x_i- x_opt)**2)
        f_error_history.append(np.max(f_value) - f_opt)

    return f_error_history


def make_communication_graph(test):  # 通信グラフを作成＆保存
    weight_graph = Communication(n, 4, 0.3)
    weight_graph.make_connected_WS_graph()
    P = weight_graph.P
    P_history = []
    for k in range(test):  # 通信グラフを作成＆保存
        weight_graph.make_connected_WS_graph()
        P_history.append(weight_graph.P)
    return P, P_history


def main_L1(n, m, lamb, pattern, test):
    p, x_opt, f_opt = optimal_L1(n, m, lamb)
    P, P_history = make_communication_graph(test)
    f_error_history = [[] for i in range(pattern)]
    for agent in range(pattern):
        f_error_history[agent] = iteration_L1(n, m, p, lamb, test, P_history, f_opt, agent)
    print('finish')

    for i in range(pattern):
        plt.plot(f_error_history[i])
    plt.yscale('log')
    plt.show()


def main_L2(n, m, lamb, pattern, test):
    p, x_opt, f_opt = optimal_L2(n, m, lamb)
    P, P_history = make_communication_graph(test)
    f_error_history = [[] for i in range(pattern)]
    for agent in range(pattern):
        f_error_history[agent] = iteration_L2(n, m, p, lamb, test, P_history, f_opt, agent)
    print('finish')

    for i in range(pattern):
        plt.plot(f_error_history[i])
    plt.yscale('log')
    plt.show()

def main_Dist(n, m, lamb, pattern, test):
    p, x_opt, f_opt = optimal_Dist(n, m, lamb)
    P, P_history = make_communication_graph(test)
    f_error_history = [[] for i in range(pattern)]
    for agent in range(pattern):
        f_error_history[agent] = iteration_Dist(n, m, p, lamb, test, P_history, f_opt, agent)
    print('finish')

    for i in range(pattern):
        plt.plot(f_error_history[i])
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    n = 20
    m = 50
    lamb = 0.1
    np.random.seed(0)  # ランダム値固定
    pattern = 2
    test = 10000
    # main_L1(n, m, lamb, pattern, test)
    # main_L2(n, m, lamb, pattern, test)
    main_Dist(n,m,lamb,pattern,test)

