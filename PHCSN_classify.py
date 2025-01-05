import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
import numpy as np
from __pycache__.config import *
import Opt_COA
from tensorflow.keras.optimizers import Adam

# convolutional siamese network (CSN)
input_layer = Input((250,250, 1))
def input1(input_layer):
    layer1 = Conv2D(16, (3,3), padding="same",activation='relu')(input_layer)
    layer1 = Conv2D(16, (3,3), padding="same",activation='relu')(layer1)
    layer2 = MaxPooling2D((2,2))(layer1)
    layer3 = Conv2D(16, (3,3), activation='relu', padding="same")(layer2)
    layer5 = Conv2D(16, (3,3), activation='relu', padding="same")(layer3)
    layer5 = MaxPooling2D((2,2))(layer5)
    layer6 = Flatten()(layer5)
    layer7 = Dense(16, activation='relu')(layer6)
    layer8 = Dense(377)(layer7)
    xx1 = layers.Dense(64, activation="relu", name="dense")(layer8)
    outputs = layers.Dense(64, activation="relu", name="dense_")(xx1)
    return outputs

def CSN_classifi():
    layer1 = Conv2D(16, (3,3), activation='relu', padding='same')(input_layer)
    layer1 = Conv2D(16, (3,3), padding="same",activation='relu')(layer1)
    layer2 = MaxPooling2D((2,2), padding='same')(layer1)
    layer3 = Conv2D(16, (3,3), activation='relu', padding='same')(layer2)
    layer3 = Conv2D(16, (3,3), activation='relu', padding='same')(layer3)
    layer4 = MaxPooling2D((2,2), padding='same')(layer3)
    layer4 = Dropout(0.2)(layer4)
    Layer5 = input1(layer4)
    outputs = layers.Dense(6, name="softmax")(xx2)
    model = Model(inputs=input_layer, outputs=outputs)  
    return  model
    
    
    
    
# Custom Parameterized Hypercomplex Convolutional Layer
class ParameterizedHypercomplexConv(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='same', activation=None, **kwargs):
        super(ParameterizedHypercomplexConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = tf.keras.activations.get(activation)

        # Learnable parameters for hypercomplex components
        self.alpha = tf.Variable(initial_value=tf.random.normal([filters]), trainable=True, name='alpha')
        self.beta = tf.Variable(initial_value=tf.random.normal([filters]), trainable=True, name='beta')
        self.gamma = tf.Variable(initial_value=tf.random.normal([filters]), trainable=True, name='gamma')
        self.delta = tf.Variable(initial_value=tf.random.normal([filters]), trainable=True, name='delta')

    def build(self, input_shape):
        # Separate convolution layers for hypercomplex components
        self.conv_r = Conv2D(self.filters, self.kernel_size, strides=self.strides, padding=self.padding)
        self.conv_i = Conv2D(self.filters, self.kernel_size, strides=self.strides, padding=self.padding)
        self.conv_j = Conv2D(self.filters, self.kernel_size, strides=self.strides, padding=self.padding)
        self.conv_k = Conv2D(self.filters, self.kernel_size, strides=self.strides, padding=self.padding)

    def call(self, inputs):
        # Split the input into four parts for hypercomplex components
        r, i, j, k = tf.split(inputs, 4, axis=-1)

        # Apply convolutions and combine with learnable parameters
        r_out = self.alpha * self.conv_r(r) + self.beta * self.conv_i(i)
        i_out = self.gamma * self.conv_j(j) + self.delta * self.conv_k(k)

        # Combine hypercomplex components
        combined_output = r_out + i_out

        # Apply activation function if specified
        if self.activation:
            combined_output = self.activation(combined_output)

        return combined_output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)


# Define the CSN Sub-Model
def input1(input_layer):
    layer1 = ParameterizedHypercomplexConv(16, (3, 3), padding="same", activation='relu')(input_layer)
    layer1 = ParameterizedHypercomplexConv(16, (3, 3), padding="same", activation='relu')(layer1)
    layer2 = MaxPooling2D((2, 2))(layer1)
    layer3 = ParameterizedHypercomplexConv(16, (3, 3), activation='relu', padding="same")(layer2)
    layer5 = ParameterizedHypercomplexConv(16, (3, 3), activation='relu', padding="same")(layer3)
    layer5 = MaxPooling2D((2, 2))(layer5)
    layer6 = Flatten()(layer5)
    layer7 = Dense(16, activation='relu')(layer6)
    layer8 = Dense(377)(layer7)
    xx1 = Dense(64, activation="relu", name="dense")(layer8)
    outputs = Dense(64, activation="relu", name="dense_")(xx1)
    return CSN_classifi()

# Parameterized Hypercomplex Convolutional Siamese network (PHCSN)
# Define the CSN Model with PHC layers
def PHCSN_model(X,Y):
    input_layer = Input((250, 250, 1))  # 4 channels for hypercomplex input
    layer1 = ParameterizedHypercomplexConv(16, (3, 3), activation='relu', padding='same')(input_layer)
    layer1 = ParameterizedHypercomplexConv(16, (3, 3), padding="same", activation='relu')(layer1)
    layer2 = MaxPooling2D((2, 2), padding='same')(layer1)
    layer3 = ParameterizedHypercomplexConv(16, (3, 3), activation='relu', padding='same')(layer2)
    layer3 = ParameterizedHypercomplexConv(16, (3, 3), activation='relu', padding='same')(layer3)
    layer4 = MaxPooling2D((2, 2), padding='same')(layer3)
    layer4 = Dropout(0.2)(layer4)
    Layer5 = input1(layer4)
    outputs = Dense(6, activation="softmax", name="softmax")(Layer5)  # 6 classes for output
    phCSN_model = Model(inputs=input_layer, outputs=outputs)
    # weight = Adam(lr=Opt_COA.optimizeweight(), decay=1e-6)
    phCSN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    phCSN_model.summary()
    history = phCSN_model.fit(X,Y,epochs=1,batch_size=32,verbose=False)
    # CSN_model.summary()
    # phCSN_model.save("model_saved/PHCSN")
    CSN_history=history.history['accuracy']
    model = assarray(CSN_history)
    print()
    print("Accuracy:", model)
    print()
    
    
    return CSN_model



