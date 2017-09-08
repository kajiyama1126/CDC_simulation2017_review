import matplotlib.pyplot as plt

from agent import new_Agent, new_Agent_moment_CDC2017_paper, new_Agent_moment_CDC2017_paper2
from iteration import new_iteration_L1


class new_iteration_L1_paper(new_iteration_L1):
    def make_agent(self, pattern):  # L1専用
        Agents = []
        s = self.step[pattern]
        for i in range(self.n):
            if pattern % 2 == 0:
                Agents.append(
                    new_Agent(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i, weight=None, R=self.R))
            elif pattern % 2 == 1:
                Agents.append(
                    new_Agent_moment_CDC2017_paper(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i,
                                                   weight=None, R=self.R))

        return Agents


class new_iteration_L1_paper2(new_iteration_L1):
    def make_agent(self, pattern):  # L1専用
        Agents = []
        s = self.step[pattern]
        for i in range(self.n):
            if pattern % 2 == 0:
                Agents.append(
                    new_Agent(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i, weight=None, R=self.R))
            elif pattern % 2 == 1:
                Agents.append(
                    new_Agent_moment_CDC2017_paper2(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i,
                                                    weight=None, R=self.R))

        return Agents

    def make_graph(self, f_error):
        label = ['DSM', 'Proposed']
        line = ['-', '-.']
        for i in range(int(self.pattern)):
            # stepsize = '_s(k)=' + str(self.step[i]) + '/k+10'
            if i == 0:
                plt.plot(f_error[i], label=label[0], linestyle=line[0], linewidth=1)
            if i % 2 == 1:
                stepsize = 'Gamma = ' + str(self.step[i])
                plt.plot(f_error[i], label=label[1] + stepsize, linestyle=line[1], linewidth=1)
        plt.legend()
        plt.yscale('log')
        plt.xlabel('iteration $k$', fontsize=10)
        plt.ylabel('$max_{i}  f(x_i(k))-f^*$',fontsize=10)
        plt.show()
