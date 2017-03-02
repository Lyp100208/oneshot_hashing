#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel

GLOG_logtostderr=0 GLOG_log_dir=./log/cifar100/usual_24/  \
./caffe/build/tools/caffe train \
    --solver=./prototxt/cifar100/usual_solver_24.prototxt \
    --gpu=1 \
    2>&1 | tee -a ./log/cifar100/usual_24/log.txt

