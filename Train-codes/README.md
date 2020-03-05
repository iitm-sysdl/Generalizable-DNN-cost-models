Usage 
=====

pip3 install -r requirements.txt 

python3 train.py --model=<name> --valdir=<name> --name=<name> --batchsize=<num> --numepochs=<num>

MODELS=mv1,mv2,mv3,resnet18,resnet34,resnet50,resnet101,resnet150,squeeze

Tensorboard Visualization
==========

The pytorch code generates a folder called runs where all tensorboard data is dumped. To invoke visualizations - run

tensorboard --logdir=/path/to/runs

Open the https:// link provided by tensorboard on your browser
