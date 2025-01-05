from tensorflow import keras
import tensorflow as tf
from keras.layers import Input, Dense,Conv2D,concatenate,Reshape,Dropout,Bidirectional,LSTM,BatchNormalization, MaxPooling2D,  Flatten
from tensorflow.keras import layers, Model
from __pycache__.config import *
import Opt_COA
from tensorflow.keras.optimizers import Adam
import keras
from keras_self_attention import SeqSelfAttention

class AxialAttention(Layer):
    def __init__(self, units, **kwargs):
        super(AxialAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W_q = self.add_weight(name="W_q", shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)
        self.W_k = self.add_weight(name="W_k", shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)
        super(AxialAttention, self).build(input_shape)

    def call(self, inputs):
        query = tf.tensordot(inputs, self.W_q, axes=[2, 0])
        key = tf.tensordot(inputs, self.W_k, axes=[2, 0])
        
        attention_width = tf.matmul(query, key, transpose_b=True)
        attention_height = tf.nn.softmax(attention_width, axis=-1)
        
        context = tf.matmul(attention_height, inputs, transpose_a=True)
        
        return context

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(AxialAttention, self).get_config()
        config['units'] = self.units
        return config
    
    

# Dilated Axial Attention convolutional neural network (DAA-CNN)
def DAA_CNN_classifi(X,Y):
    input_1 = Input(shape=(250,250, 1))
    model_1 = Conv2D(32, kernel_size=3, activation='relu')(input_1)
    model_1 = MaxPooling2D((2,2), padding='same')(model_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Conv2D(32, kernel_size=3, activation='relu')(model_1)
    model_1 = MaxPooling2D((2,2), padding='same')(model_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Conv2D(32, kernel_size=3, activation='relu')(model_1)
    model_1 = MaxPooling2D((2,2), padding='same')(model_1)
    model_1 = BatchNormalization()(model_1)
    model_2 = Conv2D(32, kernel_size=3, activation='relu')(model_1)
    model_2 = MaxPooling2D((2,2), padding='same')(model_2)
    model_2 = BatchNormalization()(model_2)
    model_2 = Conv2D(16, (3,3), activation='relu', padding='same')(model_2)
    Axia_layer = AxialAttention(Layer)
    model_3 = Flatten()(model_2)
    model_3 = Dense(6)(model_3)
    Ex_CNN_model = Model(input_1, model_3)
    weight = Adam(lr=Opt_COA.optimizeweight(), decay=1e-6)
    Ex_CNN_model.compile(optimizer= weight, loss="categorical_crossentropy",metrics=['accuracy'])
    history = Ex_CNN_model.fit(X,Y,epochs=1,batch_size=32,verbose=False)
    Ex_CNN_model.save("model_saved/Ex_CNN")
    Ex_CNN_model.summary()
    Ex_CNN_history=history.history['accuracy']
    model = assarray(Ex_CNN_history)
    print()
    print("Accuracy:", model)
    print()
    
    return model


def DAA_CNN1_classifi(X,Y):
    input_1 = Input(shape=(250,250, 1))
    model_1 = Conv2D(32, kernel_size=3, activation='relu')(input_1)
    model_1 = MaxPooling2D((2,2), padding='same')(model_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Conv2D(32, kernel_size=3, activation='relu')(model_1)
    model_1 = MaxPooling2D((2,2), padding='same')(model_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Conv2D(32, kernel_size=3, activation='relu')(model_1)
    model_1 = MaxPooling2D((2,2), padding='same')(model_1)
    model_1 = BatchNormalization()(model_1)
    model_2 = Conv2D(32, kernel_size=3, activation='relu')(model_1)
    model_2 = MaxPooling2D((2,2), padding='same')(model_2)
    model_2 = BatchNormalization()(model_2)
    model_2 = Conv2D(16, (3,3), activation='relu', padding='same')(model_2)
    Axia_layer = AxialAttention(Layer)
    model_3 = Flatten()(model_2)
    model_3 = Dense(6)(model_3)
    Ex_CNN_model = Model(input_1, model_3)
    Ex_CNN_model.compile(optimizer= 'adam', loss="categorical_crossentropy",metrics=['accuracy'])
    history = Ex_CNN_model.fit(X,Y,epochs=1,batch_size=32,verbose=False)
    Ex_CNN_model.summary()
    Ex_CNN_history=history.history['accuracy']
    model = assarray(Ex_CNN_history)
    print()
    print("Accuracy:", model)
    print()
    
    return model






































