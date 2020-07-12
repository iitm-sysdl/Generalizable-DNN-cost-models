#!/bin/bash
for i in random mutual_info_v2 spearmanCorr
do 
    python holdout.py --sampling_type=$i --learning_type=combined --name=holdout$i --numSamples=10 --model=xgb
done
