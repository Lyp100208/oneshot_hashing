#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel
 GLOG_logtostderr=0 GLOG_log_dir=./log/sun397/googlenet_support36_learnt_bilstm_48 \
./caffe/build/tools/caffe train \
    --solver=./prototxt/sun397/googlenet_support36_learnt_bilstm_slover.prototxt \
    --gpu=3 \
    --weights=./model/sun397/ \
    2>&1 | tee -a ./log/sun397/googlenet_support36_learnt_bilstm_48/log.txt

