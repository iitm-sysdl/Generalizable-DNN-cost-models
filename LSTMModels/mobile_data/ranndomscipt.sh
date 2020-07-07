for i in 5 7 10
do 
    for j in 1 2 3 .. 100
    do
        var="random_$i_samples_$j"
        python randomlstmmodel.py --sampling_type=random --learning_type=combined --name=$var --numSamples=$i --model=xgb
    done
done