import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model


def train(dataset, epochs):
    X_train = pd.read_csv(dataset)
    X_train.drop(columns='label', inplace=True)
    X_train = X_train.values.astype('float32') / 255
    X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
    input_img = Input((784,))
    encoded = Dense(units=128, activation='relu')(input_img)
    encoded = Dense(units=64, activation='relu')(encoded)
    encoded = Dense(units=32, activation='relu')(encoded)
    decoded = Dense(units=64, activation='relu')(encoded)
    decoded = Dense(units=128, activation='relu')(decoded)
    decoded = Dense(units=784, activation='sigmoid')(decoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=256, shuffle=True)
    autoencoder.save("model.h5") # will save a copy of our model in the remote server

def predict(dataset):
    X_pred = pd.read_csv(dataset)
    X_label = X_pred['label']
    X_pred.drop(columns='label', inplace=True)
    X_pred = X_pred.values.astype('float32') / 255
    X_pred = X_pred.reshape(len(X_pred), np.prod(X_pred.shape[1:]))
    autoencoder = load_model('model.h5')
    predicted = autoencoder.predict(X_pred)

    for i in range(100):
        plt.imsave("autoencoded/img{}_labael{}.png".format(i, X_label[i]), predicted[i].reshape(28, 28))

if __name__ == "__main__":
    action = sys.argv[1]
    dataset = sys.argv[2]

    if action == 'train':
        epochs = int(sys.argv[3])
        train(dataset, epochs)
    elif action == 'predict':
        predict(dataset)
