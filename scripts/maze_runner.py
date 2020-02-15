#!/usr/bin/env python

# Python imports.
import sys
import time
import argparse
import logging
import random
import pickle
# Other imports
import srl_example_setup
from simple_rl.mdp import MDPDistribution, MDPList
from simple_rl.tasks import GridWorldMDP, MazeMDP
from simple_rl.agents import QLearningAgent, TabularRMaxAgent,MultiTaskRMaxAgent, PatternLearningAgent
from simple_rl.run_experiments import run_agents_lifelong, run_agents_seq
from simple_rl.utils import make_mdp


import warnings
warnings.filterwarnings("ignore")



logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()

parser.add_argument('--run', type=int)
parser.add_argument('--height', type=int, default=4)
parser.add_argument('--width', type=int, default=4)
parser.add_argument('--mdps', type=int, default=100)
parser.add_argument('--samples', type=int, default=100)
parser.add_argument('--thres-sm', type=int, default=50)
parser.add_argument('--thres-lg', type=int, default=500)
parser.add_argument('--episodes', type=int, default=3000)
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--step-cost', type=float, default=0.2)
parser.add_argument('--pattern-gap', type=float, default=0.15)
parser.add_argument('--greedy', type=bool, default=False)
parser.add_argument('--save-dir', type=str, default='results')
parser.add_argument('--goal-w', type=int, default=4)
parser.add_argument('--goal-h', type=int, default=4)
parser.add_argument('--lava-w', type=int, default=2)
parser.add_argument('--lava-h', type=int, default=2)

args = parser.parse_args()
t0 = time.time()

if __name__ == "__main__":
    print(args)
    # Make MDP list, agents.     
    mdps = {}
    height = args.height
    width = args.width

    with open("mazes/mazefile_"+str(width)+"_"+str(height)+"_"+str(args.mdps)+"_"+str(args.samples)+"_"+str(args.run)+".pkl", "rb") as fp:   # Unpickling
        landforms = pickle.load(fp)
        print(landforms)

    mdps = []
    for k in range(args.samples):
        mdp = MazeMDP(width=width, height=height, landforms=landforms[k], is_goal_terminal=False,
                  init_loc = (1,1), goal_locs=[(args.goal_w,args.goal_h)], lava_locs=[(args.lava_w,args.lava_h)], rand_init=True, 
                  gamma=0.95, step_cost=args.step_cost)
        mdps.append(mdp)

    mdp_list = MDPList(mdps)
    

    thres_sm = args.thres_sm
    thres_lg = args.thres_lg
    
    ql_agent = QLearningAgent(actions=mdp_list.get_actions())

    rmax_agent = TabularRMaxAgent(states=mdp.states, state_map=mdp.state_map, actions=mdp.get_actions(), 
                            s_a_threshold=thres_lg, greedy=args.greedy)
    pattern_agent = PatternLearningAgent(states=mdp.states, state_map=mdp.state_map, actions=mdp.get_actions(),
                                     thres_sm=thres_sm, thres_lg=thres_lg, pattern_gap=args.pattern_gap, greedy=args.greedy)
    agents = [ql_agent, rmax_agent, pattern_agent]
    
    # Run experiment and make plot.
    run_agents_seq(agents, mdp_list, samples=args.samples,
                        episodes=args.episodes, steps=args.steps, reset_at_terminal=True, open_plot=False, dir_for_plot=args.save_dir)


