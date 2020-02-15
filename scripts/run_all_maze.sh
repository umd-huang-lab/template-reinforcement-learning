#! /bin/bash  
HEIGHT=4
WIDTH=4
EPS=5000
STP=30
RUNS=1
TASKS=100
SM=50
LG=500
MDPS=100
TASKS=100
DIR="results_maze"

for ((i=0;i<${RUNS};i++));
do
	python maze_runner.py --height ${HEIGHT} --width ${WIDTH} --episodes ${EPS} --steps ${STP} --run ${i} --thres-sm ${SM} --thres-lg ${LG} --mdps ${MDPS} --samples ${TASKS} --save-dir ${DIR} > h_${HEIGHT}_w_${WIDTH}_${DIR}.txt
	mv h_${HEIGHT}_w_${WIDTH}_${DIR}.txt ${DIR}/lifelong-maze_h-${HEIGHT}_w-${WIDTH}/h_${HEIGHT}_w_${WIDTH}.txt
    python plot.py --dir lifelong-maze_h-${HEIGHT}_w-${WIDTH} --episodes ${EPS} --samples ${TASKS} --models Q-learning RMax TempLe --type mttask --path "${DIR}/"
    python plot.py --dir lifelong-maze_h-${HEIGHT}_w-${WIDTH} --episodes ${EPS} --samples ${TASKS} --models Q-learning RMax TempLe --type mttaskrmax --path "${DIR}/"
	mv ${DIR}/lifelong-maze_h-${HEIGHT}_w-${WIDTH} ${DIR}/lifelong-maze_h-${HEIGHT}_w-${WIDTH}_run${i}
done
python plot.py --dir lifelong-maze_h-${HEIGHT}_w-${WIDTH} --episodes ${EPS} --samples ${TASKS} --models Q-learning RMax TempLe --type mttaskm --runs ${RUNS} --path "${DIR}/"
python plot.py --dir lifelong-maze_h-${HEIGHT}_w-${WIDTH} --episodes ${EPS} --samples ${TASKS} --models Q-learning RMax TempLe --type mttaskmrmax --runs ${RUNS} --path "${DIR}/"
python plot.py --dir lifelong-maze_h-${HEIGHT}_w-${WIDTH} --episodes ${EPS} --samples ${TASKS} --models Q-learning RMax TempLe --type ep2accm --runs ${RUNS} --path "${DIR}/"