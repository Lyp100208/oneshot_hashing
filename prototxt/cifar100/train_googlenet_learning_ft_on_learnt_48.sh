#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel
 GLOG_logtostderr=0 GLOG_log_dir=./log/cifar100/googlenet_learning_ft_on_learnt_48 \
./caffe/build/tools/caffe train \
    --solver=./prototxt/cifar100/googlenet_learning_ft_on_learnt_slover_48.prototxt \
    --gpu=3 \
    --weights=./model/cifar100/googlenet_support40_learnt_bilstm_48/support40_learnt48_iter_200000.caffemodel \
    2>&1 | tee -a ./log/cifar100/googlenet_learning_ft_on_learnt_48/log.txt

