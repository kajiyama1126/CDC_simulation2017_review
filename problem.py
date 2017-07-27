import cvxpy as cvx
import numpy as np

class Lasso_problem(object):
    def __init__(self, n,m,p,lamb):
        self.n = n
        self.m = m
        self.lamb = lamb
        self.p = p


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

        prob = cvx.Problem(obj)
        prob.solve()
        print(prob.status,self.x.value)

if __name__ == '__main__':
    n=10
    m=5
    p = [np.random.rand(m) for i in range(n)]
    # p = [np.array([1,1,0.1,1,0])  for i in range(n)]
    lamb = 0.1
    p_num = np.array(p)
    pro= Lasso_problem(n,m,p_num,lamb)
    pro.solve()

