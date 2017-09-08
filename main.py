import numpy as np

from paper_CDC_class import new_iteration_L1_paper,new_Agent_moment_CDC2017_paper2,new_iteration_L1_paper2

if __name__ == '__main__':
    n = 50
    m = 20
    lamb = 0.1
    R = 4
    np.random.seed(0)  # ランダム値固定
    pattern = 8
    test = 2000
    # step = [0.25, 0.25, 0.5, 0.5, 1., 1., 2., 2.]
    step = [0.5, 0.5, 0.5, 0.7, 0.5, 0.9, 0.5, 0.99]
    if pattern != len(step):
        print('error')
        pass
    else:
        # step = np.array([[0.1 *(j+1) for i in range(2)] for j in range(10)])
        step = np.reshape(step, -1)
        # tmp = new_iteration_L1_paper(n, m, step, lamb, R, pattern, test)
        tmp = new_iteration_L1_paper2(n, m, step, lamb, R, pattern, test)
