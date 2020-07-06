#!/bin/bash
for i in random mutual_info_v2 spearmanCorr
do 
    for j in slow medium fast
    do
        python hardwarecluster.py --sampling_type=$i --learning_type=combined --name=transfer$j$i --numSamples=10 --model=xgb --transfer=$j
    done
done