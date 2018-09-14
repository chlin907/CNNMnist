#!/usr/bin/env python
from CNNMnist.CNNMnist import CNNMnist
import sys, getopt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def print_usage():
    print('task_01_cnn_mnist_pipeline.py -i <input_dataset_dir>')
    print('Eg: task_01_cnn_mnist_pipeline.py -i ./data')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:m:")
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    if not opts:
        print_usage()
        sys.exit()

    for opt, arg in opts:
        if opt == '-h':
            print_usage()
            sys.exit()
        elif opt in ("-i"):
            path = arg

    # Assign CNNMnist class
    cnn_mnist = CNNMnist()
    # Load train and test datasets stored in path
    cnn_mnist.load_dataset(path)
    # Pre-process data for neural network training
    cnn_mnist.preprocess()
    # Build model
    cnn_mnist.build_model()
    # Train model
    cnn_mnist.fit(num_epoch=20, num_batch_size=64, val_split=0.2)
    # Evaluate performance on test set 
    test_loss, test_acc = cnn_mnist.test()
    print('Test result: loss = {} and acc = {}'.format(test_loss, test_acc))
    # Print out acc and loss history
    print('=========================================================================')
    print('Training accuray history: {}'.format(cnn_mnist.history.history['acc']))
    print('Validation accuray history: {}'.format(cnn_mnist.history.history['val_acc']))
    print('Training loss history: {}'.format(cnn_mnist.history.history['loss']))
    print('Validation loss history: {}'.format(cnn_mnist.history.history['val_loss']))

if __name__ == '__main__':
    main(sys.argv[1:])
