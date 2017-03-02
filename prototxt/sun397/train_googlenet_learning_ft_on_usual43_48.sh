#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel

GLOG_logtostderr=0 GLOG_log_dir=./log/sun397/googlenet_learning_ft_on_usual43_48/  \
./caffe/build/tools/caffe train \
    --solver=./prototxt/sun397/googlenet_learning_ft_on_usual43_solver_48.prototxt \
    --weights=./model/sun397/googlenet_usual43_48/usual43_iter_60000.caffemodel  \
    --gpu=1 \
    2>&1 | tee -a ./log/sun397/googlenet_learning_ft_on_usual43_48/log.txt

