#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel
GLOG_logtostderr=0 GLOG_log_dir=./log/cifar100/support_oneshot_fine_2lstm_48 \
./caffe/build/tools/caffe train \
    --solver=./prototxt/cifar100/support_oneshot_fine_2lstm_solver_48.prototxt \
    --weights=./model/cifar100/usual_48/usual48_iter_200000.caffemodel \
    --gpu=3 \
    2>&1 | tee -a ./log/cifar100/support_oneshot_fine_2lstm_48/log.txt

