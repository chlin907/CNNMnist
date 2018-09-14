import struct, os
import numpy as np

class CNNMnist:
    def __init__(self):
        self.data_dir = None
        self.is_preprocessed = False
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.model = None
        self.history = None

    def load_dataset(self, data_dir):
        # load data
        self.data_dir = data_dir

        fname_img_train = os.path.join(self.data_dir, 'train-images-idx3-ubyte')
        fname_lbl_train = os.path.join(self.data_dir, 'train-labels-idx1-ubyte')
        fname_img_test = os.path.join(self.data_dir, 't10k-images-idx3-ubyte')
        fname_lbl_test = os.path.join(self.data_dir, 't10k-labels-idx1-ubyte')

        with open(fname_lbl_train, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            self.y_train = np.fromfile(flbl, dtype=np.int8)

        with open(fname_img_train, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            self.x_train = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.y_train), rows, cols)

        with open(fname_lbl_test, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            self.y_test = np.fromfile(flbl, dtype=np.int8)

        with open(fname_img_test, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            self.x_test = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.y_test), rows, cols)

    def preprocess(self):
        from keras.utils import to_categorical
        # normalize x by 255 and add new axis (channel)
        # One-hot encoding y
        if self.is_preprocessed:
            print('Pre-processing was done before. No action is taken here')
            return
        else:
            self.is_preprocessed = True

        self.x_train = self.x_train.reshape(self.x_train.shape+(1,))
        self.x_train = self.x_train.astype('float32') / 255.0

        self.x_test = self.x_test.reshape(self.x_test.shape+(1,))
        self.x_test = self.x_test.astype('float32') / 255.0

        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    def build_model(self):
        # Build CNN model
        from keras import layers
        from keras.models import Model
        from keras.utils import plot_model

        input_layer = layers.Input(shape=(28, 28, 1))
        conv1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
        mp1 =  layers.MaxPooling2D((2, 2))(conv1)

        conv2_1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(mp1)
        mp2_1 =  layers.MaxPooling2D((2, 2))(conv2_1)
        conv3_1 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(mp2_1)

        conv2_2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(mp1)
        mp2_2 =  layers.MaxPooling2D((2, 2))(conv2_2)
        conv3_2 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(mp2_2)

        concat = layers.concatenate([conv3_1, conv3_2], axis=3)
        mp_final = layers.MaxPooling2D((2, 2))(concat)

        flat = layers.Flatten()(mp_final)
        fc1 = layers.Dense(1000, activation='relu')(flat)
        fc2 = layers.Dense(500, activation='relu')(fc1)
        output_layer = layers.Dense(10, activation='softmax')(fc2)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.summary()
        plot_model(self.model, to_file='model.png', show_shapes=True)

    def fit(self, num_epoch=2, num_batch_size=256, val_split=0.2, lr=0.001, run_callbacks=True):
        from keras import optimizers
        from keras import callbacks

        optimizer = optimizers.RMSprop(lr=lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer , metrics=['accuracy'])

        if run_callbacks:
            log_dir = './run_log'
            model_path = './weights.{epoch:02d}-{val_loss:.2f}.hdf5'
            train_callbacks = [
                callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,),
                callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=0,),
                callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')
            ]
        else:
            train_callbacks = None

        self.history = self.model.fit(self.x_train, self.y_train,
                                      epochs=num_epoch, batch_size=num_batch_size,
                                      validation_split=val_split, callbacks=train_callbacks)

    def test(self):
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        return test_loss, test_acc

    def save(self, file_path):
        model.save(file_path)
        print('Model is saved as {}'.format(file_path))

    def load_model(self, model_path):
        from keras.models import load_model
        self.model = load_model(model_path)

    def predict_proba(self, img_path):
        from keras.preprocessing import image
        img = image.load_img(img_path, grayscale=True, target_size=(28, 28))
        x = image.img_to_array(img)
        x = (255 - x).astype('float32') / 255.0

        return self.model.predict(x.reshape((1,) + x.shape))

    def get_train_set(self):
        return self.x_train, self.y_train

    def get_test_set(self):
        return self.x_test, self.y_test

