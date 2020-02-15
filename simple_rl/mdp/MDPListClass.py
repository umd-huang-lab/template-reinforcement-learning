''' MDPListClass.py: Contains the MDP List Class. '''

# Python imports.
from __future__ import print_function
import numpy as np
from collections import defaultdict

class MDPList(object):
    ''' Class for lists over MDPs. '''

    def __init__(self, mdp_list, horizon=0, is_var=False):
        '''
        Args:
            mdp_prob_dict (dict):
                Key (MDP)
                Val (float): Represents the probability with which the MDP is sampled.

        Notes:
            @mdp_prob_dict can also be a list, in which case the uniform distribution is used.
        '''

        self.num_mdps = len(mdp_list)
        self.horizon = horizon
        self.mdp_list = mdp_list
        self.is_var = is_var

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = {}
        param_dict["mdp_num"] = len(self.mdp_list)
        param_dict["horizon"] = self.horizon

        return param_dict

    def get_prob_of_mdp(self):

        return 1 / self.num_mdps

    def get_horizon(self):
        return self.horizon

    def get_actions(self):
        return self.mdp_list[0].get_actions()

    def get_num_state_feats(self):
        return self.mdp_list[0].get_num_state_feats()

    def get_gamma(self):
        '''
        Notes:
            Not all MDPs in the distribution are guaranteed to share gamma.
        '''
        return self.mdp_list[0].get_gamma()
    
    def get_mdp(self, index):
        
        return self.mdp_list[index]

    def get_reward_func(self, avg=True):
        return self.mdp_list[0].get_reward_func()

#    def get_average_reward_func(self):
#        def _avg_r_func(s, a):
#            r = 0.0
#            for m in self.mdp_list:
#                r += m.reward_func(s, a) * self.mdp_prob_dict[m]
#            return r
#        return _avg_r_func

    def get_init_state(self):
        '''
        Notes:
            Not all MDPs in the distribution are guaranteed to share init states.
        '''
        return self.mdp_list[0].get_init_state()

    def get_num_mdps(self):
        return len(self.mdp_list)

    def get_mdps(self):
        return self.mdp_list



    def set_gamma(self, new_gamma):
        for mdp in self.mdp_list:
            mdp.set_gamma(new_gamma)
#
#    def sample(self, k=1):
#        '''
#        Args:
#            k (int)
#
#        Returns:
#            (List of MDP): Samples @k mdps without replacement.
#        '''
#
#        sampled_mdp_id_list = np.random.multinomial(k, list(self.mdp_prob_dict.values())).tolist()
#        indices = [i for i, x in enumerate(sampled_mdp_id_list) if x > 0]
#
#        if k == 1:
#            return list(self.mdp_prob_dict.keys())[indices[0]]
#
#        mdps_to_return = []
#
#        for i in indices:
#            for copies in range(sampled_mdp_id_list[i]):
#                mdps_to_return.append(list(self.mdp_prob_dict.keys())[i])
#
#        return mdps_to_return
        
    def __str__(self):
        '''
        Notes:
            Not all MDPs are guaranteed to share a name (for instance, might include dimensions).
        '''
        if self.is_var:
            return "lifelong-varsize"
        return "lifelong-" + str(self.mdp_list[0])

def main():
    # Simple test code.
    from simple_rl.tasks import GridWorldMDP

    mdp_distr = {}
    height, width = 8, 8
    prob_list = [0.0, 0.1, 0.2, 0.3, 0.4]

    for i in range(len(prob_list)):
        next_mdp = GridWorldMDP(width=width, height=width, init_loc=(1, 1), goal_locs=r.sample(zip(range(1, width + 1), [height] * width), 2), is_goal_terminal=True)

        mdp_distr[next_mdp] = prob_list[i]

    m = MDPDistribution(mdp_distr)
    m.sample()

if __name__ == "__main__":
    main()
