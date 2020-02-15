#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python imports.
import random
import numpy as np
from collections import defaultdict
from simple_rl.agents.MDPGroupClass import MDPGroup
from simple_rl.agents.PatternClass import Pattern

# Local classes.
from simple_rl.agents.AgentClass import Agent

class PtTabularRMaxAgent(Agent):
    '''
    Implementation for the modified R-Max Agent with separate known thresholds for different state action pairs
    '''

    def __init__(self, states, state_map, actions, gamma=0.95, horizon=3, 
                 init_threshold=2, name="RMax", greedy=False):
        name = name
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.rmax = 1.0
        self.horizon = horizon
        self.init_threshold = init_threshold
        self.greedy = greedy
        
        self.states = states
        self.state_map = state_map
        self.actions = actions
        self.action_map = {}
        k = 0

        #Define the id of actions in the list.
        for a in self.actions:
            self.action_map[a] = k
            k += 1
#        print(self.state_map)
#        print(self.action_map)
        self.reset()

    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabula rasa config.
        '''
        self.n_s_a = np.zeros((len(self.states),len(self.actions)))
        self.n_s_a_s = np.zeros((len(self.states),len(self.actions),len(self.states)))
        self.r_s_a = np.zeros((len(self.states),len(self.actions))) # total rewards
        
        self.known_map = np.zeros((len(self.states), len(self.actions))) 
        self.threshold_map = np.ones((len(self.states), len(self.actions)))*self.init_threshold # different thresholds for different s-a pairs
        self.prev_state = None
        self.prev_action = None
    
    def set_threshold(self, s_id, a_id, thres):
        self.threshold_map[s_id][a_id] = thres
        self.known_map[s_id][a_id] = 0

    def get_num_known_sa(self):
        return np.sum(self.known_map)

    def is_known(self, s, a):
        return 1 == self.known_map[self.state_map[s]][self.action_map[a]]
    
    def is_state_known(self, s):
        '''Whether a state is known (for all actions)'''
        return len(self.actions) == np.sum(self.known_map[self.state_map[s],:])

    def act(self, state, reward):
        
        # Update given s, a, r, s' : self.prev_state, self.prev_action, reward, state
        self.update(self.prev_state, self.prev_action, reward, state)

        # Compute best action.
#        action = self.get_max_q_action(state)
        if self.is_state_known(state):
            if self.greedy:
                action = self.get_best_action(state)
            else:
                action = self.get_policy_action(state, self.q)
        else:
            action = self.balanced_wander(state)

        # Update pointers.
        self.prev_action = action
        self.prev_state = state
        
#        print("action is ", action)

        return action

    def balanced_wander(self, state):
        '''Return the action that has been taken for the least times'''
        min_action = random.choice(self.actions)
        s_id = self.state_map[state]
        min_visit_times = self.n_s_a[s_id][self.action_map[min_action]]
        
        for a in self.actions:
            if self.n_s_a[s_id][self.action_map[a]] < min_visit_times:
                min_visit_times = self.n_s_a[s_id][self.action_map[a]]
                min_action = a
        return min_action
    
    def print_policy(self):
        for s in self.states:
            print("s: ", s.get_data(), "a: ",self.get_policy_action(s, self.q))
    
    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Updates T and R.
        '''
#        print("update", state, action)
        if state != None and action != None:
            s_id = self.state_map[state]
            a_id = self.action_map[action]
            ns_id = self.state_map[next_state]
            
            if 0 == self.known_map[s_id][a_id]:
                if self.n_s_a[s_id][a_id] < self.threshold_map[s_id][a_id]:
                    # Add new data points if we haven't seen this s-a enough.
                    self.r_s_a[s_id][a_id] += reward
                    self.n_s_a[s_id][a_id] += 1
                    self.n_s_a_s[s_id][a_id][ns_id] += 1
    
                if self.n_s_a[s_id][a_id] >= self.threshold_map[s_id][a_id]:
                    self.known_map[s_id][a_id] = 1
#                    print("known", s_id, a_id, self.n_s_a_s[s_id][a_id]/self.n_s_a[s_id][a_id])
                    if not self.greedy:
                        self.q = self.planning()
                        
                    
    def get_policy_action(self, state, q):
        return self.actions[np.argmax(q[self.state_map[state]])]
            
    def get_ns_dist(self, state, action):
        return self.par_tensor[self.action_map[action]][self.state_map[state]][:len(self.states)]
    
    def get_r(self, state, action):
        return self.par_tensor[self.action_map[action]][self.state_map[state]][len(self.states)]
    
    def get_next_r(self, state, action):
        probs = self.par_tensor[self.action_map[action]][self.state_map[state]][:len(self.states)]
        rs = np.zeros((len(self.states)))
        for s in self.states:
            rs[self.state_map[s]] = max(self.par_tensor[:,self.state_map[s],len(self.states)])
        return np.dot(probs, rs)
    
    def planning(self, n_iter=10000):
        self.par_tensor = self.get_par_tensor()
        q = np.zeros((len(self.states), len(self.actions)))
        prev_q = np.copy(q)
        for i in range(n_iter):
            for s in self.states:
                for a in self.actions:
                    q[self.state_map[s]][self.action_map[a]] = self.get_r(s,a) + self.gamma * np.dot(self.get_ns_dist(s,a), np.max(q, axis=1))
            if np.linalg.norm(q-prev_q) < 1e-3:
#                print("iter for ", i, "times")
                break
            prev_q = np.copy(q)
        return q
    
    def get_best_action(self, state):
        self.par_tensor = self.get_par_tensor()
        max_a = random.choice(self.actions)
        max_q = self.get_r(state, max_a) + self.gamma * self.get_next_r(state, max_a)
        for a in self.actions:
            r = self.get_r(state, a)
            nr = self.get_next_r(state, a)
            rew = r + self.gamma * nr
            if rew > max_q:
                max_q = rew
                max_a = a
        return max_a

    def _compute_max_qval_action_pair(self, state, horizon=None):
        '''
        Args:
            state (State)
            horizon (int): Indicates the level of recursion depth for computing Q.

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon

        # Grab random initial action in case all equal
        best_action = random.choice(self.actions)
        max_q_val = self.get_q_value(state, best_action, horizon)

        # Find best action (action w/ current max predicted Q value)
        for action in self.actions:
            q_s_a = self.get_q_value(state, action, horizon)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

    def get_greedy_action(self, state):
        max_re = 0
        max_a = self.actions[0]
        for action in self.actions:
            if self.norm_rewards[state][action] > max_re:
                max_re = self.norm_rewards[state][action]
                max_a = action
        return max_a
            
    
    def get_max_q_action(self, state, horizon=None):
        '''
        Args:
            state (State)
            horizon (int): Indicates the level of recursion depth for computing Q.

        Returns:
            (str): The string associated with the action with highest Q value.
        '''

        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon 
        return self._compute_max_qval_action_pair(state, horizon)[1]

    def get_max_q_value(self, state, horizon=None):
        '''
        Args:
            state (State)
            horizon (int): Indicates the level of recursion depth for computing Q.

        Returns:
            (float): The Q value of the best action in this state.
        '''

        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon 
        return self._compute_max_qval_action_pair(state, horizon)[0]

    def get_q_value(self, state, action, horizon=None):
        '''
        Args:
            state (State)
            action (str)
            horizon (int): Indicates the level of recursion depth for computing Q.

        Returns:
            (float)
        '''

        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon

        if horizon <= 0 or state.is_terminal():
            # If we're not looking any further.
            return self._get_reward(state, action)

        # Compute future return.
        expected_future_return = self.gamma*self._compute_exp_future_return(state, action, horizon)
        q_val = self._get_reward(state, action) + expected_future_return# self.q_func[(state, action)] = self._get_reward(state, action) + expected_future_return

        return q_val

    def _compute_exp_future_return(self, state, action, horizon=None):
        '''
        Args:
            state (State)
            action (str)
            horizon (int): Recursion depth to compute Q

        Return:
            (float): Discounted expected future return from applying @action in @state.
        '''

        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon

#        next_state_dict = self.transitions[state][action]
#
#        denominator = float(sum(next_state_dict.values()))
#        state_weights = defaultdict(float)
#        for next_state in next_state_dict.keys():
#            count = next_state_dict[next_state]
#            state_weights[next_state] = (count / denominator)
        
#        weighted_future_returns = [self.get_max_q_value(next_state, horizon-1) * state_weights[next_state] for next_state in next_state_dict.keys()]

        weighted_future_returns = [self.get_max_q_value(next_state, horizon-1) * 
                                   self.norm_transitions[state][action][next_state] for next_state 
                                   in self.norm_transitions[state][action].keys()]

        return sum(weighted_future_returns)

    def _get_reward(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            Believed reward of executing @action in @state. If R(s,a) is unknown
            for this s,a pair, return self.rmax. Otherwise, return the MLE.
        '''

#        if self.r_s_a_counts[state][action] >= self.s_a_threshold:
#            # Compute MLE if we've seen this s,a pair enough.
#            rewards_s_a = self.rewards[state][action]
#            return float(sum(rewards_s_a)) / len(rewards_s_a)
        
        if self.is_known(state, action):
            return self.norm_rewards[state][action]
#        if self.is_known(state, action):
#            if self.norm_rewards[state][action] > 0.5:
#                return 1.0
#            else:
#                return 0.0
        else:
            # Otherwise return rmax.
            return self.rmax

    def get_par_tensor(self):

        '''
        using the form of (a,s,s)to represent the parameter tensor
        :return:
        '''

        s_len = len(self.states)
        shape = ((len(self.actions), s_len, s_len+1))
        para_tensor = np.zeros(shape)

        for s1 in self.states:
            s1_id = self.state_map[s1]
            for a in self.actions:
                a_id = self.action_map[a]
                if self.is_known(s1,a):
                    for s2 in self.states:
                        s2_id = self.state_map[s2]
                        para_tensor[a_id][s1_id][s2_id] = self.n_s_a_s[s1_id][a_id][s2_id] / self.n_s_a[s1_id][a_id]
                    para_tensor[a_id][s1_id][s_len] = self.r_s_a[s1_id][a_id] / self.n_s_a[s1_id][a_id]
                else:
                    for s2 in self.states:
                        s2_id = self.state_map[s2]
                        para_tensor[a_id][s1_id][s2_id] = 1 if s2_id == s1_id else 0
                    para_tensor[a_id][s1_id][s_len] = self.rmax
                    
                    
        return para_tensor
    
    def incorporate(self, s_id, a_id, pat:Pattern):
        '''
        Incorporate the counts from the pattern group
        '''
#        print("old stats ", s_id, a_id)
#        print(self.n_s_a_s[s_id][a_id]/self.n_s_a[s_id][a_id], end="// ")
#        print(self.r_s_a[s_id][a_id]/self.n_s_a[s_id][a_id])
        
        self.n_s_a[s_id][a_id] += pat.n
        # re-permute
        ori_vec = self.n_s_a_s[s_id][a_id]
        order_vec = np.array(list(range(len(self.states))))
        ori_array = np.concatenate([[ori_vec],[order_vec]],axis=0)
#        print("origin", ori_array)
#        print("permutation", ori_array[0].argsort()[::-1])
        sorted_array = ori_array[:,ori_array[0].argsort()[::-1]]
#        print("sorted", sorted_array)
        if len(sorted_array[0]) == len(pat.n_s):
            sorted_array[0] += pat.n_s
        elif len(sorted_array[0]) > len(pat.n_s):
            sorted_array[0][:len(pat.n_s)] += pat.n_s
        else:
            sorted_array[0] += pat.n_s[:len(sorted_array[0])]
#        print("added", sorted_array)
        recover_array = sorted_array[:,sorted_array[1].argsort()]
#        print("recovered", recover_array)
        self.n_s_a_s[s_id][a_id] = recover_array[0]
        
        self.r_s_a[s_id][a_id] += pat.r
        
#        print("new stats ", s_id, a_id)
#        print(self.n_s_a_s[s_id][a_id]/self.n_s_a[s_id][a_id], end="// ")
#        print(self.r_s_a[s_id][a_id]/self.n_s_a[s_id][a_id])
        
        if self.n_s_a[s_id][a_id] >= self.threshold_map[s_id][a_id]:
            self.known_map[s_id][a_id] = 1

        if not self.greedy:
            self.q = self.planning()
    
    def incorporate_group(self, group: MDPGroup):
        for s1_id in [self.state_map[s1] for s1 in self.states]:
            for a_id in [self.action_map[a] for a in self.actions]:
                self.n_s_a[s1_id][a_id] += group.n_s_a[s1_id][a_id]
                self.r_s_a[s1_id][a_id] += group.r_s_a[s1_id][a_id]
                for s2_id in [self.state_map[s2] for s2 in self.states]:
                    self.n_s_a_s[s1_id][a_id][s2_id] += group.n_s_a_s[s1_id][a_id][s2_id]

        for s_id in [self.state_map[s] for s in self.states]:
            for a_id in [self.action_map[a] for a in self.actions]:
                if self.n_s_a[s_id][a_id] > self.s_a_threshold:
                    self.known_map[s_id][a_id] = 1
            
            
            