#!/usr/bin/env sh

GLOG_logtostderr=0 GLOG_log_dir=./log/bn_1lstmb  \
./caffe/build/tools/caffe train --solver=prototxt/stack_bn_1lstmb_solver.prototxt --gpu=3
