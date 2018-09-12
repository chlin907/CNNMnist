#!/usr/bin/env bash
# Download MNIST training and testing sets http://yann.lecun.com/exdb/mnist/
#     train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
#     train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
#     t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
#     t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

# wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
# wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
# wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
# wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
