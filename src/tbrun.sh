#!/bin/bash

PS_RESULT=`ps -e | grep "tensorboard"`

DRNN_RELATIVE="D-RNN/src/logs"
GLOBAL=~/global/$DRNN_RELATIVE_LOG
SEJIN=~/sejin/$DRNN_RELATIVE_LOG
HYUNGSUN=~/hyungsun/$DRNN_RELATIVE_LOG
YONGHA=~/yongha/$DRNN_RELATIVE_LOG
KYUNGSOO=~/kyungsoo/$DRNN_RELATIVE_LOG

# Execute Tensorflow if it doesn't run
if [ -n $PS_RESULT ]
    then
        tensorboard --logdir="global:$GLOBAL,sejin:$SEJIN,hyungsun:$HYUNGSUN,yongha:$YONGHA,kyungsoo:$KYUNGSOO" --port=9898 &
        disown
    else
        echo tensorboard is already running in the background :\)
fi

# Start training if there is no option
if [ -z "$1" ]
    then
        python train.py &
        disown
    elif [ $1 = "no" ]
        then
            echo Skip training.
        else
            echo Wrong parameter! Use \"no\" to skip training.
fi
