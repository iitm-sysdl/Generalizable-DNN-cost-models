#!/bin/bash
for i in random mutual_info_v2 spearmanCorr
do 
    for j in small large giant
    do
        for k in 10
        do
            cluster="cluster"
            samples="samples"
            cluster="cluster"
            var="$k$samples$j$cluster$i"
            python networkcluster.py --sampling_type=$i --learning_type=combined --name=$var --numSamples=$k --model=xgb --networkcluster=$j
        done
    done
done