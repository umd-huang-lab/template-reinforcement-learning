# template-reinforcement-learning

Transferring knowledge among various environments is important to efficiently learn multiple tasks online. Most existing methods directly use the previously learned models or previously learned optimal policies to learn new tasks. However, these methods may be inefficient when the underlying models or optimal policies are substantially different across tasks. In this paper, we propose Template Learning (TempLe), the first PAC-MDP method for multi-task reinforcement learning that could be applied to tasks with varying state/action space. TempLe generates transition dynamics templates, abstractions of the transition dynamics across tasks, to gain sample efficiency by extracting similarities between tasks even when their underlying models or optimal policies have limited commonalities. We present two algorithms for an online and a finite-model setting respectively. We prove that our proposed TempLe algorithms achieve much lower sample complexity than single-task learners or state-of-the-art multi-task methods. We show via systematically designed experiments that our TempLe method universally outperforms the state-of-the-art multi-task methods (PAC-MDP or not) in various settings and regimes. 

## **Running Examples**

### Simple Test

The file *examples/all_examples.ipynb* provides 3 examples for various multi-task settings. The sample results are shown in the output of the code cells. 

### Complete Test

Under the folder *scripts*, there are three shell files named "run_all_maze.sh", "run_all_grid.sh", and "run_all_var.sh", which run the corresponding experiments with multiple runs and draw plots with averaged results. Use the command like

> bash run_all_maze.sh

to run them. 

Note that the number of runs are defined by the variable *RUNS* in the beginning of "run_all_xxx.sh". Please edit it before running. The default number of runs is 1. You may want to change it to any interger smaller than 30. (30 task series are pre-stored in mazes/grids/varmazes, and are enough to produce well-converged results. But you are welcome to add more tasks by following the way of generating tasks in *examples/all_examples.ipynb*)

The results will be stored in folders *results_maze* (or grid/var).  Once all runs are done, the plots of the results averaged over all runs can be found under the folder named "lifelong-xxxx_run${*RUNS*}". We already put some sample results under these 3 folders where the plots of one run are shown by the ".png" files.

## Additional Information

Our implementation was modified from [simple_rl](<https://github.com/david-abel/simple_rl/tree/master/simple_rl>). 

