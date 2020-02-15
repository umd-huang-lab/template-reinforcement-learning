''' MDPGroupClass.py: Class for an MDP group '''

# Python imports.
from collections import defaultdict
import numpy as np

class MDPGroup(object):
    ''' Abstract MDP model class. '''

    def __init__(self, agent):
        self.states = agent.states
        self.actions = agent.actions
        
        self.n_s_a = agent.n_s_a
        self.n_s_a_s = agent.n_s_a_s
        self.r_s_a = agent.r_s_a
#        self.print_group()

    def distance(self, other):
        '''
        Compare the difference between two MDPs
        '''
        if not isinstance(other, MDPGroup):
            return NotImplemented
        
        max_distance = 0
        for i in range(len(self.states)):
            for j in range(len(self.actions)):
                if self.n_s_a[i][j] > 0:
#                    dynamic = np.hstack((self.n_s_a_s[i][j] / self.n_s_a[i][j], self.r_s_a[i][j]/self.n_s_a[i][j]))
#                    other_dynamic = np.hstack((other.n_s_a_s[i][j] / other.n_s_a[i][j], other.r_s_a[i][j]/other.n_s_a[i][j]))
                    dynamic = self.n_s_a_s[i][j] / self.n_s_a[i][j]
                    reward = self.r_s_a[i][j]/self.n_s_a[i][j]
                    other_dynamic = other.n_s_a_s[i][j] / other.n_s_a[i][j]
                    other_reward = other.r_s_a[i][j]/other.n_s_a[i][j]
                    distance = np.linalg.norm(dynamic - other_dynamic) + np.linalg.norm(reward - other_reward)
                    if distance > max_distance:
                        max_distance = distance
        
        return max_distance
    
    def print_group(self, compact=False):
        print("MDP with", len(self.states), "states and", len(self.actions), "actions")
        if compact:
            print("n(s,a):")
            nzs = np.where(self.n_s_a > 0)
            for nz in np.transpose(nzs):
                print(nz, ":", self.n_s_a[nz[0]][nz[1]], end="; ")
            print("n(s,a,s'):")
            nzs = np.where(self.n_s_a_s > 0)
            for nz in np.transpose(nzs):
                print(nz, ":", self.n_s_a_s[nz[0]][nz[1]][nz[2]], end="; ")
            print("r(s,a):")
            nzs = np.where(self.r_s_a > 0)
            for nz in np.transpose(nzs):
                print(nz, ":", self.r_s_a[nz[0]][nz[1]], end="; ")
        else:
            print("n(s,a):")
            print(self.n_s_a)
            print("n(s,a,s'):")
            print(self.n_s_a_s)
            print("r(s,a):")
            print(self.r_s_a)

    def merge(self, other):
        if not isinstance(other, MDPGroup):
            return NotImplemented
        
        self.n_s_a += other.n_s_a
        self.n_s_a_s += other.n_s_a_s
        self.r_s_a += other.r_s_a

    def __str__(self):
        return str(self.name)
