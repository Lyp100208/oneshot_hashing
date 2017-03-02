#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel
GLOG_logtostderr=0 GLOG_log_dir=./log/ilsvrc2012/support_oneshot_fine_bilstm*2_48 \
./caffe/build/tools/caffe train \
    --solver=./prototxt/ilsvrc2012/support_oneshot_fine_bilstm*2_solver_48.prototxt \
    --gpu=1 \
    --snapshot=./snapshot/ilsvrc2012/support_oneshot_fine_bilstm*2_48/support_oneshot_fine_bilstm*2_m_48_iter_180000.solverstate \
    2>&1 | tee -a ./log/ilsvrc2012/support_oneshot_fine_bilstm*2_48/log_m.txt

