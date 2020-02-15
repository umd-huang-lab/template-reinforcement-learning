import os
import time
import argparse
import logging
import sys
import numpy as np

import math
import decimal
import sys
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot
import matplotlib2tikz
import numpy as np
import csv

import warnings
warnings.filterwarnings("ignore")

from matplotlib.ticker import MaxNLocator
# ax = pyplot.figure().gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
pyplot.rcParams['legend.loc'] = 'best'
# Some nice markers and colors for plotting.
markers = ['o', 's', 'D', '^', '*', 'x', 'p', '+', 'v','|']

COLOR_SHIFT = 0
color_ls = [[118, 167, 125], [102, 120, 173],\
            [198, 113, 113], [94, 94, 94],\
            [169, 193, 213], [230, 169, 132],\
            [192, 197, 182], [210, 180, 226]]
colors = [[shade / 255.0 for shade in rgb] for rgb in color_ls]
colors = colors[COLOR_SHIFT:] + colors[:COLOR_SHIFT]
linestyles = ['-', '--', ':', '-.']



def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def mtask_plot_rmax(n, models, samples, use_smooth=True):
    
#    rewards = {}
    path = args.path + args.dir
    xs = list(range(1,samples+1))
    rmax_reward = []
    with open(path+'/RMax.csv', 'r') as resFile:
        reader = csv.reader(resFile)
        data = list(reader)
        print(len(data[0]))
        for i in range(samples):
            cum = 0
            for j in range(n):
                cum += float(data[i][j])
            rmax_reward.append(cum)
    print("RMax", rmax_reward)
    
    for m, model in enumerate(models):
        model_reward = []
        with open(path+'/'+model+'.csv', 'r') as resFile:
            reader = csv.reader(resFile)
            data = list(reader)
            print(len(data[0]))
            for i in range(samples):
                cum = 0
                for j in range(n):
                    cum += float(data[i][j])
                model_reward.append(cum-rmax_reward[i])
        print(model, model_reward)
#        rewards[model] = model_reward
        if use_smooth:
            model_reward = np.array(smooth(model_reward,0.9))
            
        pyplot.plot(xs, model_reward, color=colors[m], marker=markers[m], label=model)
    
    pyplot.legend()    
    
    x_axis_label = "Tasks"
    y_axis_label = "Advantage Per-task Reward Compared with RMax"

    # Pyplot calls.
    pyplot.xlabel(x_axis_label)
    
    pyplot.ylabel(y_axis_label)
    pyplot.grid(True)
    pyplot.tight_layout() # Keeps the spacing nice.

    # Save the plot.
#     pyplot.savefig(path + "/mttask_rmax.png", format="png")
    pyplot.show()
    
    # Clear and close.
    pyplot.cla()
    pyplot.close()

def mtask_plot(n, filepath, filedir, models, samples, use_smooth=True):
    
    ax = pyplot.figure().gca()
#    rewards = {}
    path = filepath + filedir
    xs = list(range(1,samples+1))
    for m, model in enumerate(models):
        model_reward = []
        with open(path+'/'+model+'.csv', 'r') as resFile:
            reader = csv.reader(resFile)
            data = list(reader)
#             print(len(data[0]))
            for i in range(samples):
                cum = 0
                for j in range(n):
                    cum += float(data[i][j])
                model_reward.append(cum)
#         print(model, model_reward)
#        rewards[model] = model_reward
        if use_smooth:
            model_reward = np.array(smooth(model_reward,0.9))
        ax.plot(xs, model_reward, color=colors[m], marker=markers[m], label=model)
    
    ax.legend()    
    
    x_axis_label = "Tasks"
    y_axis_label = "Per-task Reward"

    # Pyplot calls.
    ax.set_xlabel(x_axis_label)
    
    ax.set_ylabel(y_axis_label)
    ax.grid(True)
#     ax.tight_layout() # Keeps the spacing nice.
    return ax
    # Save the plot.
#     pyplot.savefig(path + "/mttask.png", format="png")
#     pyplot.show()
    
#     # Clear and close.
#     pyplot.cla()
#     pyplot.close()

def multi_mtask_plot(n, models, samples, runs):
#    xs = list(range(int(gap/2),samples+1,gap))
    xs = list(range(1, samples+1))
    for m, model in enumerate(models):
        model_reward = []
        for run in range(runs):
            run_reward = []
            path = args.path + args.dir + "_run" + str(run)
            
            with open(path+'/'+model+'.csv', 'r') as resFile:
                reader = csv.reader(resFile)
                data = list(reader)
                for i in range(samples):
                    cum = 0
                    for j in range(n):
                        cum += float(data[i][j])
                    run_reward.append(cum)
            model_reward.append(run_reward)
        model_reward = np.array(model_reward)
        print(model, model_reward)
        mean_reward = np.mean(model_reward,axis=0)
        if use_smooth:
            mean_reward = np.array(smooth(mean_reward,0.9))
        pyplot.plot(xs, mean_reward, color=colors[m], label=model, marker=markers[m], markevery=int(samples/20))
#         pyplot.plot(xs, np.mean(mean_reward.reshape(-1, gap), axis=1), color=colors[m], marker=markers[m], label=model)
    
    pyplot.legend()    
    
    x_axis_label = "Tasks"
    y_axis_label = "Per-task Reward"

    # Pyplot calls.
    pyplot.xlabel(x_axis_label)
    
    pyplot.ylabel(y_axis_label)
    pyplot.grid(True)
    pyplot.tight_layout() # Keeps the spacing nice.

    # Save the plot.
#     pyplot.savefig(path + "/mttask_runs"+str(runs)+".png", format="png")
    pyplot.show()
    
    # Clear and close.
    pyplot.cla()
    pyplot.close()
    
def multi_mtask_plot_rmax(n, models, samples, runs, with_abs=False, use_smooth=True):
#    xs = list(range(int(gap/2),samples+1,gap))
    xs = list(range(1, samples+1))
    
    rmax_reward = []
    for run in range(runs):
        run_reward = []
        path = args.path + args.dir + "_run" + str(run)
        
        with open(path+'/RMax.csv', 'r') as resFile:
            reader = csv.reader(resFile)
            data = list(reader)
            for i in range(samples):
                cum = 0
                for j in range(n):
                    cum += float(data[i][j])
                run_reward.append(cum)
        rmax_reward.append(run_reward)
    rmax_reward = np.array(rmax_reward)
    print("rmax", rmax_reward)
    rmax_mean = np.mean(rmax_reward,axis=0)
        
    for m, model in enumerate(models):
        model_reward = []
        for run in range(runs):
            run_reward = []
            path = args.path + args.dir + "_run" + str(run)
            
            with open(path+'/'+model+'.csv', 'r') as resFile:
                reader = csv.reader(resFile)
                data = list(reader)
                for i in range(samples):
                    cum = 0
                    for j in range(n):
                        cum += float(data[i][j])
                    run_reward.append(cum)
            model_reward.append(run_reward)
        model_reward = np.array(model_reward)
        print(model, model_reward)
        mean_reward = np.mean(model_reward,axis=0)
        if use_smooth:
            mean_reward = np.array(smooth(mean_reward-rmax_mean,0.9))
#         pyplot.plot(xs, np.mean(mean_reward.reshape(-1, gap), axis=1)-np.mean(rmax_mean.reshape(-1, gap), axis=1), 
#                     color=colors[m], marker=markers[m], label=model)
        pyplot.plot(xs, mean_reward, color=colors[m], label=model, marker=markers[m], markevery=5)
    
    if with_abs:
        abs_models = ["Q-learningQPhiEpsilon", "Q-learningQPhid", "Q-learningQPhidaa"]
        model_names = ["Q-learning$-\phi_{Q_\epsilon^*}$", "Q-learning$-\phi_{Q_d^*}$", "Q-learning-$\phi_{Q^*}$"]
        for m, model in enumerate(abs_models):
            model_reward = []
            for run in range(runs):
                run_reward = []
                path2 = "results_maze_large/h_4_w_4_run" + str(run) + "/" + args.dir

                with open(path2+'/'+model+'.csv', 'r') as resFile:
                    reader = csv.reader(resFile)
                    data = list(reader)
                    for i in range(samples):
                        cum = 0
                        for j in range(n):
                            cum += float(data[i][j])
                        run_reward.append(cum)
                model_reward.append(run_reward)
            model_reward = np.array(model_reward)
        #     print(model, model_reward)
            mean_reward = np.mean(model_reward,axis=0)
            if use_smooth:
                mean_reward = np.array(smooth(mean_reward-rmax_mean,0.9))
#             pyplot.plot(xs, np.mean(mean_reward.reshape(-1, gap), axis=1)-np.mean(rmax_mean.reshape(-1, gap), axis=1), 
#                         color=colors[m+len(models)], marker=markers[m+len(models)], label=model_names[m])
            pyplot.plot(xs, mean_reward, color=colors[len(models)+2], label=model_names[m], ls=linestyles[m])
    
    pyplot.legend()    
    
    x_axis_label = "Tasks"
    y_axis_label = "Advantage Per-task Reward Compared with RMax"

    # Pyplot calls.
    pyplot.xlabel(x_axis_label)
    
    pyplot.ylabel(y_axis_label)
    pyplot.grid(True)
    pyplot.tight_layout() # Keeps the spacing nice.

    # Save the plot.
#     if with_abs:
#         pyplot.savefig(path + "/mttaskrmax_runs"+str(runs)+"_abs.png", format="png")
#     else:
#         pyplot.savefig(path + "/mttaskrmax_runs"+str(runs)+".png", format="png")
    pyplot.show()
    
    # Clear and close.
    pyplot.cla()
    pyplot.close()

def multi_acc_plot(n, models, samples, runs, with_abs=False, shade=False):
    xs = list(range(1,n+1))
    
    for m, model in enumerate(models):
        model_rewards = []
        for run in range(runs):
            path = args.path + args.dir + "_run" + str(run)
            with open(path+'/'+model+'.csv', 'r') as resFile:
                reader = csv.reader(resFile)
                data = list(reader)
                
                for i in range(samples):
                    eps_rewards = []
                    cum = 0
                    for j in range(n):
                        cum += float(data[i][j])
                        eps_rewards.append(cum)
                        
                    model_rewards.append(eps_rewards)
        model_rewards = np.array(model_rewards)
#        print(model, model_rewards)
        mean_reward = np.mean(model_rewards,axis=0)
        
        pyplot.plot(xs, mean_reward, markevery=max(int(len(mean_reward) / 30), 1),
                    color=colors[m], marker=markers[m], label=model)

    if with_abs:
        abs_models = ["Q-learningQPhiEpsilon", "Q-learningQPhid", "Q-learningQPhidaa"]
        model_names = ["Q-learning$-\phi_{Q_\epsilon^*}$", "Q-learning$-\phi_{Q_d^*}$", "Q-learning-$\phi_{Q^*}$"]
        for m, model in enumerate(abs_models):
            model_rewards = []
            for run in range(runs):
                path2 = "results_maze_large/h_4_w_4_run" + str(run) + "/" + args.dir
                with open(path2+'/'+model+'.csv', 'r') as resFile:
                    reader = csv.reader(resFile)
                    data = list(reader)
                    
                    for i in range(samples):
                        eps_rewards = []
                        cum = 0
                        for j in range(n):
                            cum += float(data[i][j])
                            eps_rewards.append(cum)
                            
                        model_rewards.append(eps_rewards)
            model_rewards = np.array(model_rewards)
    #        print(model, model_rewards)
            mean_reward = np.mean(model_rewards,axis=0)
            
            pyplot.plot(xs, mean_reward, markevery=max(int(len(mean_reward) / 30), 1),
                        color=colors[m+len(models)], marker=markers[m+len(models)], label=model_names[m])
    
    pyplot.legend()    
    
    x_axis_label = "Episodes"
    y_axis_label = "Cumulative Reward"

    # Pyplot calls.
    pyplot.xlabel(x_axis_label)
    
    pyplot.ylabel(y_axis_label)
    pyplot.grid(True)
    pyplot.tight_layout() # Keeps the spacing nice.

    # Save the plot.
#     pyplot.savefig(path + "/mean_acc_runs"+str(runs)+".png", format="png")
    pyplot.show()
    
    # Clear and close.
    pyplot.cla()
    pyplot.close()

