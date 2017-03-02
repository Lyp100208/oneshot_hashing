#!/usr/bin/env sh
#-weights bvlc_googlenet.caffemodel
#-weights ~/code/shell/models/bvlc_alexnet.caffemodel
./caffe/build/tools/caffe train \
    --solver=./prototxt/cifar100/cifar100_solver_24.prototxt \
    --gpu=2  
