#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel

GLOG_logtostderr=0 GLOG_log_dir=./log/sun397/googlenet_usual36_48/  \
./caffe/build/tools/caffe train \
    --solver=./prototxt/sun397/googlenet_usual36_solver_48.prototxt \
    --weights=/home/libing/tmp/bvlc_googlenet.caffemodel  \
    --gpu=1 \
    2>&1 | tee -a ./log/sun397/googlenet_usual36_48/log1.txt

