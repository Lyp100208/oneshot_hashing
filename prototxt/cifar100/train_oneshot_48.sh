#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel

GLOG_logtostderr=0 GLOG_log_dir=./log/cifar100/oneshot_48/  \
./caffe/build/tools/caffe train \
    --solver=./prototxt/cifar100/oneshot_solver_48.prototxt \
    --gpu=3 \
    --weights=./snapshot/cifar100/usual_48/usual48_iter_200000.caffemodel \
    2>&1 | tee -a ./log/cifar100/oneshot_48/log.txt

