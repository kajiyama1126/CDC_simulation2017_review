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
        subgrad_l1 = self.lamb*np.sign(self.x_i)
        subgrad = grad + subgrad_l1
        return subgrad

    def send(self):
        return self.x_i, self.name

    def receive(self, x_j, name):
        self.x[name] = x_j

    def s(self, k):
        return 2.0/ (k + 10.0)

    def update(self, k):
        self.x[self.name] = self.x_i
        self.x_i = np.dot(self.weight, self.x)
        self.x_i = self.x_i- self.s(k) * self.subgrad()

class Agent_L2(Agent):
    def subgrad(self):
        grad = self.x_i-self.p
        grad_l2 = 2*self.lamb * self.x_i
        return grad+grad_l2

class Agent_Dist(Agent):
    def subgrad(self):
        grad = (self.x_i-self.p)/np.linalg.norm((self.x_i-self.p),2)
        return grad


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

        self.v_i = self.gamma * np.dot(self.weight, self.v) + self.s(k)*(0.1) * self.subgrad()
        self.x_i = np.dot(self.weight, self.x) - self.v_i

class Agent_moment_CDC2017_L2(Agent_moment_CDC2017):
    def subgrad(self):
        grad = self.x_i-self.p
        grad_l2 = 2*self.lamb * self.x_i
        return grad+grad_l2

class Agent_moment_CDC2017_Dist(Agent_moment_CDC2017):
    def subgrad(self):
        grad = (self.x_i-self.p)/np.linalg.norm((self.x_i-self.p),2)
        return grad

class Agent_moment_CDC2017_s(Agent_moment_CDC2017):
    def update(self, k):
        self.x[self.name] = self.x_i
        self.v[self.name] = self.v_i

        self.v_i = self.gamma *self.s(k)/self.s(k-1)*np.dot(self.weight, self.v) + self.s(k)*(0.2) * self.subgrad()
        self.x_i = np.dot(self.weight, self.x) - self.v_i

class Agent_harnessing(Agent):
    def __init__(self, n, m, p, lamb, name, weight=None):
        super(Agent_harnessing, self).__init__(n, m, p, lamb, name, weight)

        self.v_i = self.subgrad()
        self.v = np.zeros([self.n, self.m])
        self.eta = 0.01

    def update(self, k):
        self.x[self.name] = self.x_i
        self.v[self.name] = self.v_i
        grad_bf = self.subgrad()
        self.x_i = np.dot(self.weight, self.x) - self.v_i
        self.v_i = np.dot(self.weight, self.v) + self.eta*( self.subgrad() -grad_bf)
