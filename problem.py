import cvxpy as cvx
import numpy as np

class Lasso_problem(object):
    def __init__(self, n,m,p,lamb,R):
        self.n = n
        self.m = m
        self.lamb = lamb
        self.p = p
        self.R = R


    def solve(self):
        n,m =self.n,  self.m
        self.x = cvx.Variable(m)
        f_1 = []
        for i in range(n):
            f_1.append((1 / 2 * cvx.power(cvx.norm((self.x - self.p[i]), 2),2)))

        f_2 = self.lamb * n * cvx.norm(self.x,1)
        obj = cvx.Minimize(0)
        for i in range(n):
            obj += cvx.Minimize(f_1[i])
        obj += cvx.Minimize(f_2)
        const = [cvx.norm(self.x,2)<=self.R]
        self.prob = cvx.Problem(obj,const)
        self.prob.solve()
        print(self.prob.status,self.x.value)

    def send_f_opt(self):
        return self.prob.value

class Ridge_problem(object):
    def __init__(self, n,m,p,lamb,R):
        self.n = n
        self.m = m
        self.lamb = lamb
        self.p = p
        self.R = R

    def solve(self):
        n,m =self.n,  self.m
        self.x = cvx.Variable(m)
        f_1 = []
        for i in range(n):
            f_1.append(1 / 2 * cvx.power(cvx.norm((self.x - self.p[i]), 2),2))

        f_2 = self.lamb * n * cvx.power(cvx.norm(self.x,2),2)
        obj = cvx.Minimize(0)
        for i in range(n):
            obj += cvx.Minimize(f_1[i])
        obj += cvx.Minimize(f_2)
        const = [cvx.norm(self.x,2)<=self.R]
        self.prob = cvx.Problem(obj,const)
        self.prob.solve()
        print(self.prob.status,self.x.value)

    def send_f_opt(self):
        return self.prob.value

class Dist_problem(object):
    def __init__(self, n,m,p,lamb,R):
        self.n = n
        self.m = m
        self.lamb = lamb
        self.p = p
        self.R = R

    def solve(self):
        n,m =self.n,  self.m
        self.x = cvx.Variable(m)
        f_1 = []
        for i in range(n):
            f_1.append(cvx.norm((self.x - self.p[i]), 2))

        obj = cvx.Minimize(0)
        for i in range(n):
            obj += cvx.Minimize(f_1[i])

        const = [cvx.norm(self.x,2)<=self.R]
        self.prob = cvx.Problem(obj,const)
        self.prob.solve()
        print(self.prob.status,self.x.value)

    def send_f_opt(self):
        return self.prob.value


if __name__ == '__main__':
    n=10
    m=5
    p = [np.random.rand(m) for i in range(n)]
    # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
    lamb = 0.1
    p_num = np.array(p)
    pro= Lasso_problem(n,m,p_num,lamb)
    pro.solve()

