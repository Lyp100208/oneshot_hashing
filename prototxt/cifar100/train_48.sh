#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel
GLOG_logtostderr=0 GLOG_log_dir=./log/cifar100/all_48/ \
./caffe/build/tools/caffe train \
    --solver=./prototxt/cifar100/48_solver.prototxt \
    --snapshot=./snapshot/cifar100/all_48/all48_iter_160000.solverstate \
    --gpu=1  \
    2>&1 | tee  -a ./log/cifar100/all_48/log.txt
