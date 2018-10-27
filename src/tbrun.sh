#!/bin/bash

PS_RESULT=`ps -e | grep "tensorboard"`
if [ -n $PS_RESULT ]
    then
        tensorboard --logdir="logs" --port=9898 &
        disown
    else
        echo tensorboard is already running in the background :\)
fi

#echo "=========================================================================="
#echo "Enter the commands below to keep running if the ssh connection is closed."
#echo "$ Ctrl + Z"
#echo "$ bg"
#echo "$ disown"
#echo "=========================================================================="
python train.py &
disown
