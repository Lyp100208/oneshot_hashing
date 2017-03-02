#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel

GLOG_logtostderr=0 GLOG_log_dir=./log/cifar100/googlenet_usual90_48/  \
./caffe/build/tools/caffe train \
    --solver=./prototxt/cifar100/googlenet_usual90_solver_48.prototxt \
    --snapshot=./snapshot/cifar100/googlenet_usual90_48/usual90_48_inner_iter_50000.solverstate \
    --gpu=0 \
    2>&1 | tee -a ./log/cifar100/googlenet_usual90_48/log.txt

