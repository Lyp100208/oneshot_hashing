#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel
 GLOG_logtostderr=0 GLOG_log_dir=./log/ilsvrc2012/googlenet_support40_learnt_bilstm_48 \
./caffe/build/tools/caffe train \
    --solver=./prototxt/ilsvrc2012/googlenet_support40_learnt_bilstm_slover.prototxt \
    --gpu=0 \
    --weights=./model/ilsvrc2012/googlenet_usual_48/googlenet_usual48_iter_200000.caffemodel \
    2>&1 | tee -a ./log/ilsvrc2012/googlenet_support40_learnt_bilstm_48/log.txt

