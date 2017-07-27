import numpy as np


class Agent(object):
    def __init__(self, n, m, p, lamb, name, weight=None):
        self.n = n
        self.m = m
        self.p = p
        self.lamb = lamb
        self.name = name
        self.weight = weight

        self.x_i = np.random.rand(self.m)
        self.x = np.zeros([self.n, self.m])

    def subgrad(self):
        grad = self.x_i - self.p
        # grad = np.zeros_like(self.x_i)
        subgrad_l1 = np.zeros(self.m)
        for i in range(len(self.x_i)):
            if self.x_i[i] > 0:
                subgrad_l1[i] = self.lamb
            elif self.x_i[i] < 0:
                subgrad_l1[i] = -self.lamb
            else:
                subgrad_l1[i] = 0

        subgrad = grad + subgrad_l1
        return subgrad

    def send(self):
        return self.x_i, self.name

    def receive(self, x_j, name):
        self.x[name] = x_j

    def s(self, k):
        return 1.0/ (k + 10.0)

    def update(self, k):
        self.x[self.name] = self.x_i
        self.x_i = np.dot(self.weight, self.x) - self.s(k) * self.subgrad()


class Agent_moment_CDC2017(Agent):
    def __init__(self, n, m, p, lamb, name, weight=None):
        super(Agent_moment_CDC2017, self).__init__(n, m, p, lamb, name, weight)
        self.gamma = 0.9

        self.v_i = self.subgrad()
        self.v = np.zeros([self.n, self.m])

    def send(self):
        return (self.x_i, self.v_i), self.name

    def receive(self, x_j, name):
        self.x[name] = x_j[0]
        self.v[name] = x_j[1]

    def update(self, k):
        self.x[self.name] = self.x_i
        self.v[self.name] = self.v_i

        self.v_i = self.gamma * np.dot(self.weight, self.v) + self.s(k)*(0.2) * self.subgrad()
        self.x_i = np.dot(self.weight, self.x) - self.v_i

class Agent_moment_CDC2017_s(Agent_moment_CDC2017):
    def update(self, k):
        self.x[self.name] = self.x_i
        self.v[self.name] = self.v_i

        self.v_i = self.gamma *self.s(k)/self.s(k-1)*np.dot(self.weight, self.v) + self.s(k)*(0.2) * self.subgrad()
        self.x_i = np.dot(self.weight, self.x) - self.v_i