#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel

./caffe/build/tools/caffe train \
    --solver=./prototxt/cifar100/support_oneshot_solver_48.prototxt \
    --gpu=0 \
    --weights=./model/cifar100/usual_48/usual48_iter_200000.caffemodel

