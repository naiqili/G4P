#!/bin/bash

maxiter=500
subn=20
lb=5
M=50
inverse='True'
ggd='True'
matid='-2 0.85'


matcnt=$(echo ${matid} | wc -w)
for dataset in synthetic1 synthetic2 synthetic3 synthetic4 synthetic5 synthetic6
do
    for i in `seq 0 9`;
    # for i in 3;
    do
        echo "phase $i"
        python main.py --dataset ${dataset} --phase $i --maxiter ${maxiter} --subn ${subn} --lb ${lb} --inverse ${inverse} --matid ${matid} --M ${M} --ggd ${ggd} --gpu 0  > ./log/main3_dataset${dataset}_inv${inverse}_ggd${ggd}_M${M}_iter${maxiter}_mat_pearson_subn${subn}_lb5_s5_log${i}.log
    done
done