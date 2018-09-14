# CNNMnist

CNNMnist is a mini-project to apply convolutional neural network to [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. The whole flow contains two parts: 

- Pipeline for model training
- Script to classify an input image by a trained model

## Usage

- Model training can be started by executing 
  *task_01_cnn_mnist_pipeline.py -i <input_dataset_dir>*
- Digit prediction of an input image can by done by executing 
  *task_02_cnn_mnist_predict.py -i <input_image> -m <trained_model>*

Demonstraition of the steps above can be found in demo_task_01_cnn_mnist_pipeline.ipynb and demo_task_02_cnn_mnist_predict.ipynb. 

Details about the model setup and modules of pipeline step can be found in class [CNNMnist](https://github.com/chlin907/CNNMnist/tree/master/CNNMnist).



## CNN Model



<img src="./doc/model.png" alt="drawing" width="600"/>

## Requirement

- Numpy
- Keras and Tensorflow