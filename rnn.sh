#!/bin/bash

source activate conda_env_2.7

subject_min=0
subject_max=116


for j in `seq $subject_min $subject_max`
do
	echo "THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python rnn.py $j" > RNN$j.sh
	qsub -cwd -o jobout$j -e joberr$j RNN$j.sh
	echo "RNN on subject $j "
done


rm RNN*.sh
