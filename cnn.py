from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
from keras.callbacks import TensorBoard

import csv
import cv2
import glob
import numpy as np

from keras.models import model_from_yaml


# def load_data():

def get_im(path):
    # Load as grayscale
    img = cv2.imread(path, 0)
    # Reduce size
    return img
    # resized = cv2.resize(img, (128, 96))
    # return resized


def load_train(img_names, img_class, folder_path):
    num_classes = 9
    X = []
    # y = np.zeros((len(img_class), num_classes))
    y = []

    print('Read train images')

    for i in range(len(img_names)):

        # Load label
        label = img_class[i].replace('(', '').replace(')', '')
        label = label.split(', ')
        vertical_angle = int(label[0])
        horizontal_angle = int(label[1])
        # print(vertical_angle, horizontal_angle)
        new_row = [0] * num_classes
        if vertical_angle in [30, -30] or horizontal_angle in [30, -30]:
            continue
        if vertical_angle >= 30:
            a = 0
        elif vertical_angle > -30:
            a = 1
        else:
            a = 2
        if horizontal_angle > 30:
            b = 0
        elif horizontal_angle >= -30:
            b = 1
        else:
            b = 2
        label = a * 3 + b
        # print(label)
        # y[i, label] = 1
        new_row[label] = 1
        # print(new_row)
        # print(y)
        y.append(new_row)

        # Load data
        path = folder_path + ("/%s.jpg" % img_names[i])
        img = cv2.imread(path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (200, 200))
        X.append(img)
    # print(np.array(X).shape, np.array(y).shape)
    return np.array(X), np.array(y)


def build_model(input_shape):
    num_classes = 9
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    return model


def load_model(path_to_model, model_name='model'):
    # load YAML and create model
    # yaml_file = open(path_to_model+'/model.yaml', 'r')
    yaml_file = open('%s/%s.yaml' % (path_to_model, model_name), 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    # loaded_model.load_weights(path_to_model+"/model.h5")
    loaded_model.load_weights('%s/%s.h5' % (path_to_model, model_name))
    print("Loaded model")
    return loaded_model


def save_model(model, path_to_model, model_name='model'):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open('%s/%s.yaml' % (path_to_model, model_name), 'w') as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights('%s/%s.h5' % (path_to_model, model_name))
    print("Saved model to disk")


# Read the .csv file
def read_data(path):
    img_name = []
    img_class = []
    with open(path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            img_name.append(row[0])
            img_class.append(row[1])
    return img_name, img_class


def main():
    batch_size = 32
    num_classes = 9
    epochs = 10

    # input image dimensions
    img_x, img_y = 200, 200

    (x_train, y_train) = load_train(img_name[0:2000], img_class[0:2000], 'data/cropped')
    print(x_train.shape)
    print(y_train.shape)
    # print(y_train)
    (x_test, y_test) = load_train(img_name[2000:-1], img_class[2000:-1], 'data/cropped')
    print(x_test.shape)
    # x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
    # x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
    model = build_model((img_x, img_y, 3))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                              write_graph=True, write_images=False)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard])

    save_model(model, 'models', model_name='model2')
    # evaluate loaded model on test data
    loaded_model = load_model("models")
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


if __name__ == '__main__':
    # main()
    img_name, img_class = read_data('data/classification_angle.csv')
    main()
