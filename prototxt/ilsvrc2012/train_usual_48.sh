#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel

GLOG_logtostderr=0 GLOG_log_dir=./log/ilsvrc2012/usual_48/  \
./caffe/build/tools/caffe train \
    --solver=./prototxt/ilsvrc2012/usual_solver_48.prototxt \
    #--snapshot=./snapshot/ilsvrc2012/usual_48/usual48_iter_40000.solverstate \
    --gpu=1 \
    2>&1 | tee -a ./log/ilsvrc2012/usual_48/log.txt

