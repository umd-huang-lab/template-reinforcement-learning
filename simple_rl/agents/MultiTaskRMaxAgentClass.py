#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#MultiTaskR

"""
MultiTasRMaxAgentClass is based on the TabularRMaxAgentClass for each task
"""

# Python imports.
from collections import defaultdict

# Local classes.
from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.TabularRMaxAgentClass import TabularRMaxAgent
from simple_rl.agents.MDPGroupClass import MDPGroup
import numpy as np
import math


class MultiTaskRMaxAgent(Agent):
    '''
     Implementation for the multi-task RL agent
    '''

    def __init__(self, states, state_map, actions, gamma=0.95, horizon=3, name="FMRL",
                 thres_sm=5, thres_lg=10, t1=6, model_gap=0.4, greedy=False, xi=0.2):
        name = name
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)

        self.horizon = horizon
        self.changed1 = False
        self.changed2 = False
        self.thres_sm = thres_sm
        self.thres_lg = thres_lg
        self.epsilon = 0.3
        self.states = states
        self.state_map = state_map
        self.greedy = greedy
        self.t1 = t1
        self.xi = xi
        self.model_gap = model_gap
        self.has_incorp = False  # for phase 2: whether has already incorporated past groups
        self.single_agent = None
        self.groups = []
        self.flag = []
        self.count = -1  # how many tasks we have learned
        self.reset()

    def end_of_episode(self):
        '''
        Summary:
            Resets the agents prior pointers.
        '''
        self.prev_state = None
        self.prev_action = None
        self.episode_number += 1
        self.single_agent.end_of_episode()
        
    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabula rasa config.
        '''
        if self.single_agent is not None and (self.count+1) < self.t1:
            self.groups.append(MDPGroup(self.single_agent))
            print("\n number of groups", len(self.groups))

        if self.flag != None and (self.count+1) > self.t1:
            print(self.flag)

        # start from a small threshold
        self.single_agent = TabularRMaxAgent(self.states, self.state_map, self.actions, self.gamma,
                                             s_a_threshold=self.thres_sm, greedy=self.greedy)
        self.step_number = 0
        self.has_incorp = False

        self.cur_thres = self.thres_sm

        self.count += 1  # how many episodes are completed
        print("renew the single agent")

        if (self.count+1) == self.t1:
            # the end of the first phase
            self.groups = self.grouping()
            print("The number of candidate groups: ", len(self.groups))

            self.candidate_groups = self.groups
            self.hatc = len(self.candidate_groups)
            self.c = np.zeros((len(self.candidate_groups), len(self.candidate_groups)))  # cij
            self.Delta = np.zeros((len(self.candidate_groups), len(self.candidate_groups)))  # deltaij
            self.flag = []
            for g in range(len(self.candidate_groups)):
                self.flag.append(1)

        if (self.count+1) > self.t1:
            self.flag = []
            for i in range(len(self.candidate_groups)):
                self.flag.append(1)
            self.hatc = len(self.candidate_groups)
            self.c = np.zeros((len(self.candidate_groups), len(self.candidate_groups)))  # cij
            self.Delta = np.zeros((len(self.candidate_groups), len(self.candidate_groups)))  # deltaij


    def act(self, state, reward):

        self.changed1 = False
        self.changed2 = False
        if self.count < self.t1:  # phase 1 learning

            if self.cur_thres == self.thres_sm and self.single_agent.get_num_known_sa() == len(self.states) * len(
                    self.actions):
                # reset to known threshold to be large
                print("reset threshold")
                self.single_agent.reset_thres(self.thres_lg)
                self.cur_thres = self.thres_lg

            action = self.single_agent.act(state, reward)

        if self.count >= self.t1:  # phase 2 learning

            Known1 = self.single_agent.get_num_known_sa()

            if not self.has_incorp:
                if(self.single_agent.prev_state != None and self.single_agent.prev_action != None):

                    self.filter_group(state, reward)

                    '''remember to change the threshold'''
                    addition = 0
                    for i in range(len(self.flag)):
                        addition += self.flag[i]
                    if addition == 1:
#                        print(self.flag)
                        target = 0
                        for j in range(len(self.flag)):
                            if self.flag[j] == 1:
                                target = j
                        print("this task belongs to group ", target)
                        self.single_agent.reset_thres(self.thres_lg)
                        self.single_agent.incorporate(self.candidate_groups[target])
#                        print(self.single_agent.r_s_a)
                        self.has_incorp = True
                        
#            if self.cur_thres == self.thres_sm and self.single_agent.get_num_known_sa() == len(self.states) * len(
#                    self.actions):
#                self.single_agent.reset_thres(self.thres_lg)
#                self.cur_thres = self.thres_lg

            action = self.single_agent.act(state, reward)

            Known2 = self.single_agent.get_num_known_sa()

            if(Known1 != Known2):
                self.changed1 = True

            if self.changed1 == True or self.changed2 == True:
                self.single_agent.q = self.single_agent.planning()

            self.step_number += 1

        self.prev_action = action
        self.prev_state = state

        return action

    def grouping(self):
        print("-------grouping--------")
        res_groups = []
        for group in self.groups:
            if len(res_groups) == 0:
                res_groups.append(group)
            else:
                flag = False
                for obj_group in res_groups:
                    if group.distance(obj_group) <= self.model_gap:
                        obj_group.merge(group)
                        flag = True
                if not flag:
                    res_groups.append(group)

        for i, g in enumerate(res_groups):
            print("group ", i)
            g.print_group()

        return res_groups

    
    def filter_group(self, state, reward):

        prev_action = self.single_agent.prev_action
        prev_state = self.single_agent.prev_state
        current_vector = np.zeros(len(self.states) + 1)
        current_vector[self.state_map[state]] += 1
        current_vector[len(self.states)] += reward
        dynamics = []
        dynamics_new = []
        Intervals = np.zeros(len(self.candidate_groups)) #l2 confidence intervals
        l1norm = 0 # l1norm from different c in C
        l1norm_i = 0 # l1norm from i to z
        l1norm_j = 0 # l1norm from j to z
        maxthetac = 0
        xi = self.xi
        delta = 0.01


        s_id = self.single_agent.state_map[prev_state]
        a_id = self.single_agent.action_map[prev_action]

        for i in range(len(self.candidate_groups)):
            current_group = self.candidate_groups[i]
            dynamics.append(np.hstack((current_group.n_s_a_s[self.state_map[prev_state]][self.single_agent.action_map[prev_action]]\
                       /current_group.n_s_a[self.state_map[prev_state]][self.single_agent.action_map[prev_action]],\
                        current_group.r_s_a[self.state_map[prev_state]][self.single_agent.action_map[prev_action]]\
                       /current_group.n_s_a[self.state_map[prev_state]][self.single_agent.action_map[prev_action]])))


        for k in range(len(self.candidate_groups)):
            if self.flag[k] == 1 and self.single_agent.n_s_a[s_id][a_id] != 0:
                Intervals[k] = np.sqrt((2/self.single_agent.n_s_a[s_id][a_id])*(math.log(2/delta, math.e)+2))
            else:
                Intervals[k] = 0

        maxthetac = Intervals[0]
        for g in range(len(self.candidate_groups)-1):
            if maxthetac < Intervals[g+1]:
                maxthetac = Intervals[g+1]

        for i in range(0, len(self.flag)):
            for j in range(i+1, len(self.flag)):
                if self.flag[j] == 1 and self.flag[i] == 1:
                    absolute = abs(dynamics[i] - dynamics[j])
                    absolute_i = abs(dynamics[i] - current_vector)
                    absolute_j = abs(dynamics[j] - current_vector)

                    for ele1 in absolute:
                        l1norm += ele1
                    for ele2 in absolute_i:
                        l1norm_i += ele2
                    for ele3 in absolute_j:
                        l1norm_j += ele3

                    if l1norm >= 8 * maxthetac:
                        self.c[i][j] += (1 / 4) * np.square(l1norm)
                        self.Delta[i][j] += np.square(l1norm_i) - np.square(l1norm_j)
#                        print("c:", self.c)
                        if self.c[i][j] >= xi:
                            print("This is count ", self.count+1)
                            print("This is step number ", self.step_number)
                            if self.Delta[i][j] >= 0:
                                self.flag[i] = 0
                            else:
                                self.flag[j] = 0

        for i in range(len(self.flag)):
            if self.flag[i] == 1:
                dynamics_new.append(dynamics[i])

        if len(dynamics_new) > 1:
            confirmlist = []
            for j in range(0, len(dynamics_new)):
                for k in range(j + 1, len(dynamics_new)):
                    total = 0
                    vec = dynamics_new[j] - dynamics_new[k]
                    for element in vec:
                        total += np.square(element)
                    if total > self.epsilon:
                        confirmlist.append(1)
            if sum(confirmlist) == 0:
                self.changed2 = True
'''
        for i in range(len(self.flag)):
            dynamics = []
            if self.flag[i] == 1:
                dynamics.append(dynamics[i])
                #Compute all the possible dynamics used in Check_Known procedure.

        changed = False
        for i in range(len(self.single_agent.states)):
            for j in range(len(self.single_agent.actions)):
'''


    






