#!/usr/bin/env python
from CNNMnist.CNNMnist import CNNMnist
import numpy as np
import sys, getopt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def print_usage():
    print('task_02_cnn_mnist_predict.py -i <input_image> -m <trained_model>')
    print('Eg: task_02_cnn_mnist_predict.py -i ./test.png -m ./model.h5')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:m:")
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    if not opts:
        print_usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print_usage()
            sys.exit()
        elif opt in ("-i"):
            img_path = arg
        elif opt in ("-m"):
            model_path = arg

    # Assign trained model and input image paths
    #model_path = './trained_model/weights.04-0.07.hdf5'
    #img_path = './doc/test_2.png'

    # Load model 
    cnn_mnist = CNNMnist()
    cnn_mnist.load_model(model_path)
    # Predict probability of digig 0 - 9 outcomes
    proba = cnn_mnist.predict_proba(img_path)
    # Output the digit with highest probability
    print('=========================================================================')
    print('Digit in "{}" is recognized as {} with probability = {:.2f} %'.format(img_path, np.argmax(proba), 100 * np.max(proba)))

if __name__ == '__main__':
    main(sys.argv[1:])
