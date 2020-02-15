#! /bin/bash  
HEIGHT=4
WIDTH=4
EPS=5000
STP=30
RUNS=1
MDPS=2
TASKS=50
T1=10
SM=100
LG=500
DIR="results_grid"

for ((i=0;i<${RUNS};i++));
do
	python grid_runner.py --run ${i} --height ${HEIGHT} --width ${WIDTH} --episodes ${EPS} --steps ${STP} --samples ${TASKS} --t1 ${T1} --mdps ${MDPS} --thres-sm ${SM} --thres-lg ${LG} --save-dir ${DIR} > grid_h_${HEIGHT}_w_${WIDTH}_${DIR}.txt
	mv grid_h_${HEIGHT}_w_${WIDTH}_${DIR}.txt ${DIR}/lifelong-gridworld_h-${HEIGHT}_w-${WIDTH}/grid_h_${HEIGHT}_w_${WIDTH}.txt
	mv ${DIR}/lifelong-gridworld_h-${HEIGHT}_w-${WIDTH} ${DIR}/lifelong-gridworld_h-${HEIGHT}_w-${WIDTH}_run${i}
done
python plot.py --dir lifelong-gridworld_h-${HEIGHT}_w-${WIDTH} --episodes ${EPS} --samples ${TASKS} --models Q-learning RMax TempLe FMRL --type mttaskm --runs ${RUNS} --path "${DIR}/"
python plot.py --dir lifelong-gridworld_h-${HEIGHT}_w-${WIDTH} --episodes ${EPS} --samples ${TASKS} --models Q-learning RMax TempLe FMRL --type mttaskmrmax --runs ${RUNS} --path "${DIR}/"
python plot.py --dir lifelong-gridworld_h-${HEIGHT}_w-${WIDTH} --episodes ${EPS} --samples ${TASKS} --models Q-learning RMax TempLe FMRL --type ep2accm --runs ${RUNS} --path "${DIR}/"