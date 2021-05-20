#!/bin/bash

maxiter=2000
subn=100
lb=5
M=10
inverse='True'
ggd='True'
matid='-2 0.95'
dataset='MSCI'

matcnt=$(echo ${matid} | wc -w)
for i in `seq 0 6`;
# for i in 3;
do
    echo "phase $i"
    python main3.py --dataset ${dataset} --phase $i --maxiter ${maxiter} --subn ${subn} --lb ${lb} --inverse ${inverse} --matid ${matid} --M ${M} --ggd ${ggd} --gpu 1  > ./log/main3_dataset${dataset}_inv${inverse}_ggd${ggd}_M${M}_iter${maxiter}_mat_pearson_subn${subn}_lb5_s5_log${i}.log
done
