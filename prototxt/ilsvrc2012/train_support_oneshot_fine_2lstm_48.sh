#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel
GLOG_logtostderr=0 GLOG_log_dir=./log/ilsvrc2012/support_oneshot_fine_2lstm_48 \
./caffe/build/tools/caffe train \
    --solver=./prototxt/ilsvrc2012/support_oneshot_fine_2lstm_solver_48.prototxt \
    --weights=./model/ilsvrc2012/usual_48/usual48_iter_200000.caffemodel \
    --gpu=0 \
    2>&1 | tee -a ./log/ilsvrc2012/support_oneshot_fine_2lstm_48/log_m.txt

