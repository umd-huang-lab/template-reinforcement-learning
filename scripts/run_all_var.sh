#! /bin/bash  
EPS=5000
STP=30
RUNS=1
SM=100
LG=500
TASKS=80
DIR="results_var"

for ((i=0;i<${RUNS};i++));
do
	python var_runner.py --episodes ${EPS} --steps ${STP} --run ${i} --thres-sm ${SM} --thres-lg ${LG} --samples ${TASKS} --save-dir ${DIR} > ${DIR}.txt
	mv ${DIR}.txt ${DIR}/lifelong-varsize/var.txt
    python plot.py --dir lifelong-varsize --episodes ${EPS} --samples ${TASKS} --models Q-learning RMax TempLe --type mttask --path "${DIR}/"
    python plot.py --dir lifelong-varsize --episodes ${EPS} --samples ${TASKS} --models Q-learning RMax TempLe --type mttaskrmax --path "${DIR}/"
	mv ${DIR}/lifelong-varsize ${DIR}/lifelong-varsize_run${i}
done
python plot.py --dir lifelong-varsize --episodes ${EPS} --samples ${TASKS} --models Q-learning RMax TempLe --type mttaskm --runs ${RUNS} --path "${DIR}/"
python plot.py --dir lifelong-varsize --episodes ${EPS} --samples ${TASKS} --models Q-learning RMax TempLe --type mttaskmrmax --runs ${RUNS} --path "${DIR}/"
python plot.py --dir lifelong-varsize --episodes ${EPS} --samples ${TASKS} --models Q-learning RMax TempLe --type ep2accm --runs ${RUNS} --path "${DIR}/"