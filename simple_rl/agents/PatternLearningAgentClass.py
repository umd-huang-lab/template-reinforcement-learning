#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PatternLearning is based on the PtTabularRMaxAgentClass for each and every 
"""

# Python imports.
from collections import defaultdict

# Local classes.
from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.PtTabularRMaxAgentClass import PtTabularRMaxAgent
from simple_rl.agents.MDPGroupClass import MDPGroup
from simple_rl.agents.PatternClass import Pattern
import numpy as np
import math


class PatternLearningAgent(Agent):
    '''
     Implementation for the template learning RL agent
    '''

    def __init__(self, states, state_map, actions, gamma=0.95, horizon=3, name="TempLe",
                 thres_sm=5, thres_lg=10, pattern_gap=0.4, greedy=True, 
                 with_grouping=False, t1=0, model_gap=0.4, flag_tol=3):
        name = name
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)

        self.horizon = horizon
        self.thres_sm = thres_sm
        self.thres_lg = thres_lg
        self.states = states
        self.state_map = state_map
        self.greedy = greedy
        self.pattern_gap = pattern_gap
        self.patterns = []
        self.single_agent = None
        self.with_grouping = with_grouping
        
        if self.with_grouping:
            self.t1 = t1
            self.groups = []
            self.model_gap = model_gap
            self.flag_tol = flag_tol
        
        self.count = 0
        self.reset()
    
    def reset_with_states(self, states, state_map):
        if self.single_agent is not None:
            print(len(self.patterns))
            for s_id in range(len(self.states)):
                for a_id in range(len(self.actions)):
                    p_id = self.pattern_map[s_id][a_id]
                    if p_id != -1:
                        new_pat = Pattern(self.single_agent, s_id, a_id)
                        self.patterns[int(p_id)].merge(new_pat)
            print("visits:")
            print(self.single_agent.n_s_a)
            print("pattern_map:")
            print(self.pattern_map)
            print("unknowns",len(np.where(self.pattern_map==-1)[0]), "/", len(self.states)*len(self.actions))
            for p_id,pat in enumerate(self.patterns):
                print("pattern ", p_id)
                pat.print_pattern()
            self.min_gap()
            # if number of mdp groups is known
            if self.with_grouping:
                if self.count <= self. t1:
                    self.groups.append(MDPGroup(self.single_agent))
                    print("\n number of groups:", len(self.groups))
                
                if self.count == self.t1: # end of phase 1
                    self.groups = self.grouping()
                    print("The number of candidate groups: ", len(self.groups))
                    self.patterns, self.pmap_list, self.perm_list = self.refine_pattern()
                    for p_id,pat in enumerate(self.patterns):
                        print("pattern ", p_id)
                        pat.print_pattern()
                        
        self.states = states
        self.state_map = state_map
        self.count += 1
        # start from a small threshold
        self.single_agent = PtTabularRMaxAgent(self.states, self.state_map, self.actions, self.gamma,
                                             init_threshold=self.thres_sm, greedy=self.greedy)
        self.known_map = np.zeros((len(self.states), len(self.actions)))
        self.pattern_map = np.ones((len(self.states), len(self.actions))) * (-1)
        
        if self.with_grouping and self.count > self.t1: # phase 2
            self.identify_flags = np.ones(len(self.groups)) * self.flag_tol
            self.p2_flag = False

    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabula rasa config.
        '''
        # If last task is completed, merge the identified patterns into the pattern library
        if self.single_agent is not None:
            print(len(self.patterns))
            for s_id in range(len(self.states)):
                for a_id in range(len(self.actions)):
                    p_id = self.pattern_map[s_id][a_id]
                    if p_id != -1:
                        new_pat = Pattern(self.single_agent, s_id, a_id)
                        self.patterns[int(p_id)].merge(new_pat)
            print("visits:")
            print(self.single_agent.n_s_a)
            print("pattern_map:")
            print(self.pattern_map)
            print("unknowns",len(np.where(self.pattern_map==-1)[0]), "/", len(self.states)*len(self.actions))
            for p_id,pat in enumerate(self.patterns):
                print("pattern ", p_id)
                pat.print_pattern()
            self.min_gap()
            # if number of mdp groups is known
            if self.with_grouping:
                if self.count <= self. t1:
                    self.groups.append(MDPGroup(self.single_agent))
                    print("\n number of groups:", len(self.groups))
                
                if self.count == self.t1: # end of phase 1
                    self.groups = self.grouping()
                    print("The number of candidate groups: ", len(self.groups))
                    self.patterns, self.pmap_list, self.perm_list = self.refine_pattern()
                    for p_id,pat in enumerate(self.patterns):
                        print("pattern ", p_id)
                        pat.print_pattern()
        
        self.count += 1
        # start from a small threshold
        self.single_agent = PtTabularRMaxAgent(self.states, self.state_map, self.actions, self.gamma,
                                             init_threshold=self.thres_sm, greedy=self.greedy)
        self.known_map = np.zeros((len(self.states), len(self.actions)))
        self.pattern_map = np.ones((len(self.states), len(self.actions))) * (-1)
        
        if self.with_grouping and self.count > self.t1: # phase 2
            self.identify_flags = np.ones(len(self.groups)) * self.flag_tol
            self.p2_flag = False
        
    def min_gap(self):
        listi = []
        listj = []
        for i in range(len(self.patterns)):
            for j in range(i+1, len(self.patterns)):
#                print(i,j, ":", end=" ")
                pati = self.patterns[i]
                patj = self.patterns[j]
                dist = pati.distance(patj)
                print(dist)
                if dist < self.pattern_gap:
                    listi.append(i)
                    listj.append(j)
#        print(listi, listj)
        new_patterns = []
        for k in range(len(listi)):
            self.patterns[listi[k]].merge(self.patterns[listj[k]])
        for i in range(len(self.patterns)):
            if i not in listj:
                new_patterns.append(self.patterns[i])
        self.patterns = new_patterns
#        print("new num of patterns", len(self.patterns))
#        for p_id,pat in enumerate(self.patterns):
#            print("pattern ", p_id)
#            pat.print_pattern()
        
    def act(self, state, reward):
        
        action = self.single_agent.act(state, reward)
        self.update()
        
        self.prev_action = action
        self.prev_state = state

        return action
    
    def update(self):
        cur_known_map = self.single_agent.known_map
        update_loc = np.where((cur_known_map - self.known_map)==1)
        
#        print("cur ", cur_known_map)
#        print("self ", self.known_map)
#        print(update_loc)
        
        # If there is an undated known pair
        if len(update_loc[0]) > 0:
            
            s_id = update_loc[0][0]
            a_id = update_loc[1][0]
            if self.single_agent.threshold_map[s_id][a_id] == self.thres_sm:
                self.single_agent.set_threshold(s_id, a_id, self.thres_lg)
                new_pat = Pattern(self.single_agent, s_id, a_id)
#                print("new pair:", s_id,a_id)
#                print(self.single_agent.n_s_a_s[s_id][a_id])
#                new_pat.print_pattern()
                # choose the pattern with the smallest distance and the distance is smaller than gap threshold
                best_match = -1
                best_dist = len(self.states) * 2
#                print("for", s_id, a_id)
                for p_id, pat in enumerate(self.patterns):
                    dist = pat.distance(new_pat)
                    if dist <= self.pattern_gap:
#                        print("dist:",dist,"best:", best_dist, end=" ")
                        if dist <= best_dist:
                            best_dist = dist
                            best_match = p_id
#                            break
                if best_match != -1:
                    best_pat = self.patterns[best_match]
#                    print("merge to")
#                    best_pat.print_pattern()
                    best_pat.merge(new_pat)
                    self.single_agent.incorporate(s_id,a_id,best_pat)
                    self.pattern_map[s_id][a_id] = best_match
                else:
#                    print("create")
                    self.patterns.append(new_pat) 
                    self.pattern_map[s_id][a_id] = len(self.patterns)-1
                   
#                print("current pattern map")
#                print(self.pattern_map)
                if self.with_grouping and self.count > self.t1 and not self.p2_flag:
                    self.filter_group(s_id, a_id)
                    
            elif self.single_agent.threshold_map[s_id][a_id] == self.thres_lg:
                p_id = self.pattern_map[s_id][a_id]
                new_pat = Pattern(self.single_agent, s_id, a_id)
                self.patterns[int(p_id)].merge(new_pat)
             
            self.known_map[s_id][a_id] = 1

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

#        for i, g in enumerate(res_groups):
#            print("group ", i)
#            g.print_group()

        return res_groups
    
    def refine_pattern(self):
        patterns = []
        pmap_list = []
        perm_list = []
        for i, group in enumerate(self.groups):
            print("group ", i)
            pmap = np.ones((len(self.states), len(self.actions))) * (-1)
            for s_id in range(len(self.states)):
                for a_id in range(len(self.actions)):
                    new_pat = Pattern(group, s_id, a_id)
                    best_match = -1
                    best_dist = len(self.states) * 2
                    for p_id, pat in enumerate(patterns):
                        dist = pat.distance(new_pat)
                        if dist <= self.pattern_gap:
    #                        print("dist:",dist,"best:", best_dist, end=" ")
                            if dist <= best_dist:
                                best_dist = dist
                                best_match = p_id
    #                            break
                    if best_match != -1:
                        pmap[s_id][a_id] = best_match
                    else:
    #                    print("create")
                        patterns.append(new_pat) 
                        pmap[s_id][a_id] = len(patterns)-1
            pmap_list.append(pmap)
        print("all pmaps", pmap_list)
        
        return patterns, pmap_list, perm_list
    
    def filter_group(self, s_id, a_id):
        '''
        remove groups with different patterns
        '''
#        print("filtering")
        for i, group in enumerate(self.groups):
            if self.pmap_list[i][s_id][a_id] != self.pattern_map[s_id][a_id]:
                self.identify_flags[i] -= 1
        
        if len(np.where(self.identify_flags>0)[0]) == 1:
            print(self.identify_flags)
            ind = np.argmax(self.identify_flags)
            print("identified as", ind)
#            self.single_agent.incorporate_group(self.groups[ind])
            for s in range(len(self.states)):
                for a in range(len(self.actions)):
                    self.single_agent.set_threshold(s, a, self.thres_lg)
#                    print("pat", s,a,int(self.pmap_list[ind][s][a]))
                    self.single_agent.incorporate(s,a,self.patterns[int(self.pmap_list[ind][s][a])])
                    self.pattern_map[s][a] = int(self.pmap_list[ind][s][a])
                    if self.single_agent.known_map[s][a] == 1:
                        self.known_map[s][a] = 1 
            self.p2_flag = True
#            print(self.pattern_map)
            print("known", self.known_map)
    
    def end_of_episode(self):
        '''
        Summary:
            Resets the agents prior pointers.
        '''
        self.prev_state = None
        self.prev_action = None
        self.episode_number += 1
        self.single_agent.end_of_episode()
        
    def get_par_tensor(self):
        return self.single_agent.get_par_tensor()