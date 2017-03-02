#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel
 GLOG_logtostderr=0 GLOG_log_dir=./log/ilsvrc2012/support_learning_from_learnt_48 \
./caffe/build/tools/caffe train \
    --solver=./prototxt/ilsvrc2012/support_learning_from_learnt_solver_48.prototxt \
    --gpu=2 \
    --weights=./model/ilsvrc2012/support_learnt_48/support_learnt48_iter_200000.caffemodel \
    2>&1 | tee -a ./log/ilsvrc2012/support_learning_from_learnt_48/log.txt

