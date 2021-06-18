import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras import Model

class CNN(Model):
    
        def __init__(self):
                super(CNN, self).__init__()
                
                self.conv1 = Conv2D(32, 3, padding='same', activation='relu')
                self.conv2 = Conv2D(64, 3, padding='same', activation='relu')
                
                self.pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')
                self.dropout1 = Dropout(0.25)
                
                self.flatten = Flatten()
                self.d1 = Dense(128, activation='relu')
                self.dropout2 = Dropout(0.5)
                self.d2 = tf.keras.layers.Dense(3, activation='softmax')
                
                
        def call(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.pool1(x)
                x = self.dropout1(x)
                x = self.flatten(x)
                x = self.d1(x)
                x = self.dropout2(x)
                x = self.d2(x)
                return x

        def model(self):
            x = keras.Input(shape=(15, 15, 1))
            return Model(inputs=[x], outputs=self.call(x))
