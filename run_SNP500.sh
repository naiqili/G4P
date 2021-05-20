#!/bin/bash

# maxiter=20
# subn=100
# lb=5
# M=10
# inverse='True'
# ggd='True'
# matid='-1 0.85'

# matcnt=$(echo ${matid} | wc -w)
# # for i in `seq 0 11`;
# for i in 10;
# do
#     echo "phase $i"
#     python main3.py --phase $i --maxiter ${maxiter} --subn ${subn} --lb ${lb} --inverse ${inverse} --matid ${matid} --M ${M} --ggd ${ggd} --gpu 1  > ./log/main2_noval_inv${inverse}_ggd${ggd}_M${M}_iter${maxiter}_mat_pearson_subn${subn}_lb5_s5_log${i}.log
# done

# maxiter=20
# subn=100
# lb=5
# M=10
# inverse='True'
# ggd='False'
# matid='-2 0.85'

# matcnt=$(echo ${matid} | wc -w)
# # for i in `seq 0 11`;
# for i in 10;
# do
#     echo "phase $i"
#     python main3.py --phase $i --maxiter ${maxiter} --subn ${subn} --lb ${lb} --inverse ${inverse} --matid ${matid} --M ${M} --ggd ${ggd} --gpu 1  > ./log/main2_noval_inv${inverse}_ggd${ggd}_M${M}_iter${maxiter}_mat_spearman_subn${subn}_lb5_s5_log${i}.log
# done

maxiter=2000
subn=100
lb=5
M=10
inverse='True'
ggd='True'
matid='-3 0.7'

matcnt=$(echo ${matid} | wc -w)
for i in `seq 0 11`;
# for i in 10;
do
    echo "phase $i"
    python main3.py --phase $i --maxiter ${maxiter} --subn ${subn} --lb ${lb} --inverse ${inverse} --matid ${matid} --M ${M} --ggd ${ggd} --gpu 1  > ./log/main2_noval_inv${inverse}_ggd${ggd}_M${M}_iter${maxiter}_mat_kandell_subn${subn}_lb5_s5_log${i}.log
done

# mat coding:
# -1: pearson
# -2: spearman
# -3: kandell
# 48 35 83 25 84 19 21 71 24 68
# 81 38 51 56 77 43 80 15 30 60
#matid='48 35 83 25 84 19 21 71 24 68 81 38 51 56 77 43 80 15 30 60 69 37 73 62 34 54 57 70 39 42 50 55 63 28 61 65 26 72 49 64 14 23 36 53 66 76 16'