# -*- coding: utf-8 -*-
import numpy as np
from agent import Agent,Agent_moment_CDC2017,Agent_moment_CDC2017_s
from make_communication import Communication
from problem import Lasso_problem
import matplotlib.pylab as plt

n = 20
m = 5
np.random.seed(1)#ランダム値固定
p = [np.random.rand(m) for i in range(n)]
# p = [np.array([1,1,0.1,1,0])  for i in range(n)]
lamb = 0.1
p_num = np.array(p)
# np.reshape(p)
prob = Lasso_problem(n,m,p_num,lamb)
prob.solve()
x_opt = np.array(prob.x.value)#最適解
x_opt = np.reshape(x_opt,(-1,))#reshape
print(x_opt)

weight_graph = Communication(n,2,0.3)
weight_graph.make_connected_WS_graph()
P = weight_graph.P
Agents = []
pattern = 3
x_error_history = [[] for i in range(pattern)]
x_moment_error_history = []
test = 10000

P_history = []
for k in range(test):
    weight_graph.make_connected_WS_graph()
    P_history.append(weight_graph.P)

for agent in range(pattern):
    Agents = []
    for i in range(n):
        if agent == 0:
            Agents.append(Agent(n, m, p[i], lamb, name=i, weight=P[i]))
        elif agent == 1:
            Agents.append(Agent_moment_CDC2017(n, m, p[i], lamb, name=i, weight=P[i]))
        elif agent == 2:
            Agents.append(Agent_moment_CDC2017_s(n, m, p[i], lamb, name=i, weight=P[i]))
    for k in range(test):
        #グラフの時間変化
        for i in range(n):
            Agents[i].weight = P_history[k][i]

        for i in range(n):
            for j in range(n):
                x_i,name = Agents[i].send()
                Agents[j].receive(x_i,name)

        for i in range(n):
            Agents[i].update(k)

        # x_ave = 0
        # for i in range(n):
        #     x_ave += 1.0/n * Agents[i].x_i

        x_error_history[agent].append(np.linalg.norm(Agents[0].x_i- x_opt)**2)
# for i in range(n):
#     print(Agents[i].x_i)
for i in range(pattern):
    plt.plot(x_error_history[i])
plt.yscale('log')
plt.show()
