''' MDPGroupClass.py: Class for an MDP group '''

# Python imports.
from collections import defaultdict
import numpy as np

class Pattern(object):
    ''' Abstract dynamics class. '''

    def __init__(self, agent, s_id, a_id):
        
        self.n = agent.n_s_a[s_id][a_id]  # scalar
        self.n_s = self.permute(agent.n_s_a_s[s_id][a_id])  # vector
        self.r = agent.r_s_a[s_id][a_id]  # scalar

    def permute(self, vec):
        return -np.sort(-vec)
    
    def print_pattern(self):
        print("total visits: ", self.n)
        print("dynamics:", self.n_s / self.n)
        print("rewards:",self.r / self.n)
        
    def distance(self, other):
        '''
        Compare the difference between two MDPs
        '''
        if not isinstance(other, Pattern):
            return NotImplemented
        
#        dynamic = np.hstack((self.n_s / self.n, self.r/self.n))
#        other_dynamic = np.hstack((other.n_s / other.n, other.r/other.n))
#        distance = np.linalg.norm(dynamic - other_dynamic)
        dynamic = self.n_s / self.n
        other_dynamic = other.n_s / other.n
        reward = self.r / self.n
        other_reward = other.r / other.n
        
        if len(dynamic) == len(other_dynamic):
            distance = np.linalg.norm(dynamic - other_dynamic) + np.linalg.norm(reward - other_reward)
        
        elif len(dynamic) > len(other_dynamic):
            temp = np.zeros(len(dynamic))
            temp[:len(other_dynamic)] = other_dynamic
            distance = np.linalg.norm(dynamic - temp) + np.linalg.norm(reward - other_reward)
            
        else:
            temp = np.zeros(len(other_dynamic))
            temp[:len(dynamic)] = dynamic
            distance = np.linalg.norm(temp - other_dynamic) + np.linalg.norm(reward - other_reward)
        
        return distance
    

    def merge(self, other):
        if not isinstance(other, Pattern):
            return NotImplemented
        if self.n > 1e10: # sample size large enough, do not need to merge
            return
        self.n += other.n
#        print(self.n_s)
#        print(other.n_s)
        if len(self.n_s) == len(other.n_s):
            self.n_s += other.n_s
        elif len(self.n_s) > len(other.n_s):
            temp = np.zeros(len(self.n_s))
            temp[:len(other.n_s)] = other.n_s
            self.n_s += temp
        else:
            temp = np.zeros(len(other.n_s))
            temp[:len(self.n_s)] = self.n_s
            self.n_s = temp + other.n_s
        self.r += other.r

    def __str__(self):
        return str(self.name)



