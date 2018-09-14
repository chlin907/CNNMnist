# class CNNMnist

| API                                                          | Function                                    | Input                                                        | Output                          |
| ------------------------------------------------------------ | ------------------------------------------- | :----------------------------------------------------------- | ------------------------------- |
| load_dataset(path)                                           | Load dataset                                | path <str>: directory path of MNIST dataset                  | None                            |
| preprocess()                                                 | Preprocess data                             | None                                                         | None                            |
| build_model()                                                | Build model                                 | None                                                         | None                            |
| fit(num_epoch, num_batch_size, val_split, lr, run_callbacks) | Train model                                 | num_epoch <int>: # of epochs. num_batch_size <int>: batch size. val_split <float>: val. set ratio. lr <float>: learning rate. run_callbacks <bool>: run callbacks | None                            |
| test()                                                       | Evaluate performance on test set            | None                                                         | <tuple>: (Loss, acc)            |
| save(model_path)                                             | Save most recent model                      | model_path <str>: path to save model                         | None                            |
| load_model(model_path)                                       | Load pre-trained model                      | model_path <str>: path to load model                         | None                            |
| predict_proba(img_path)                                      | Predict the probability for 10 digits (0~9) | img_path <str>: path of input image                          | <numpy array>: array of size 10 |



