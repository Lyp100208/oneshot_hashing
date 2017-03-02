#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel

GLOG_logtostderr=0 GLOG_log_dir=./log/ilsvrc2012/googlenet_usual_48/  \
./caffe/build/tools/caffe train \
    --solver=./prototxt/ilsvrc2012/googlenet_usual_solver_48.prototxt \
    --weights=/home/libing/tmp/bvlc_googlenet.caffemodel \
    --gpu=0 \
    2>&1 | tee -a ./log/ilsvrc2012/googlenet_usual_48/log.txt

