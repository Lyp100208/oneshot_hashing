#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel

GLOG_logtostderr=0 GLOG_log_dir=./log/cifar100/googlenet_oneshot_ft_on_usual_48/  \
./caffe/build/tools/caffe train \
    --solver=./prototxt/cifar100/googlenet_oneshot_ft_on_usual_solver_48.prototxt \
    --weights=./model/cifar100/googlenet_usual_48/usual48_inner_iter_180000.caffemodel  \
    --gpu=1 \
    2>&1 | tee -a ./log/cifar100/googlenet_oneshot_ft_on_usual_48/log.txt

