#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel
 GLOG_logtostderr=0 GLOG_log_dir=./log/cifar100/support_learning_fine_48 \
./caffe/build/tools/caffe train \
    --solver=./prototxt/cifar100/support_learning_fine_solver_48.prototxt \
    --gpu=3 \
    --weights=./model/cifar100/usual_48/usual48_iter_200000.caffemodel \
    2>&1 | tee -a ./log/cifar100/support_learning_fine_48/log.txt

