import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Flatten, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization,
    MultiHeadAttention, Activation, Lambda, Layer
)
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from __pycache__.config import *
import Opt_COA


def SSA_TNet_Classifi():
    input_1 = Input(shape=(250,250, 1))
    model_1 = MaxPooling2D((2,2), padding='same')(input_1)
    model_1 = AveragePooling2D(pool_size=(2, 2),strides=(1, 1), padding='same')(model_1)
    model_1 = Dropout(0.2)(model_1)
    model_2 = Dense(16, activation='relu')(model_1)
    model_2 = Dense(32, activation='relu')(model_2)
    model_2 = Dense(32,  activation="relu")(model_2)
    model_2 = Activation(activation='relu')(model_2)
    # Encoding1
    layer = BatchNormalization()(model_2)
    encoding_1 = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))
    encoding_1 = encoding_1(layer, layer)
    encoding_1 = BatchNormalization()(encoding_1)
    encoding_1 = Activation(activation='relu')(encoding_1)
    encoding_1 = Dense(64)(encoding_1)
    # Encoding2
    layer1 = BatchNormalization()(encoding_1)
    encoding_2 = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))
    encoding_2 = encoding_2(layer1, layer1)
    encoding_2 = BatchNormalization()(encoding_2)
    encoding_2 = Activation(activation='relu')(encoding_2)
    encoding_2 = Dense(16)(encoding_2)
    # Encoding3
    
    layer2 = BatchNormalization()(encoding_2)
    encoding_3 = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))
    encoding_3 = encoding_3(layer2, layer2)
    encoding_3 = BatchNormalization()(encoding_3)
    encoding_3 = Flatten()(encoding_3)
    encoding_3 = Activation(activation='relu')(encoding_3)
    encoding_3 = Dense(16,  activation="softmax")(encoding_3)
    encoding_3 = Dense(6)(encoding_3)
    SSA_TNet_model = Model(input_1, encoding_3)
    
    return SSA_TNet_model


# Relational Transformer Layer
class RelationalTransformer(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, **kwargs):
        super(RelationalTransformer, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.attention_layers = [MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim) for _ in range(num_layers)]
        self.feedforward_layers = [Dense(ff_dim, activation='relu') for _ in range(num_layers)]

    def call(self, inputs):
        x = inputs
        for i in range(self.num_layers):
            attention_output = self.attention_layers[i](x, x)
            x = BatchNormalization()(attention_output + x)  # Add residual connection
            x = self.feedforward_layers[i](x)
            x = BatchNormalization()(x)
        return x


# Wrapper for tf.expand_dims to handle KerasTensor
class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)


# SSA_TNet1 with Relational Transformer
def SSA_TNet1_with_RelationalTransformer(input_shape):
    input_1 = Input(shape=input_shape)

    # Initial convolutional-like structure
    model_1 = MaxPooling2D((2, 2), padding='same')(input_1)
    model_1 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(model_1)
    model_1 = Dropout(0.2)(model_1)
    model_2 = Dense(16, activation='relu')(model_1)
    model_2 = Dense(32, activation='relu')(model_2)
    model_2 = Dense(32, activation="relu")(model_2)
    model_2 = Activation(activation='relu')(model_2)

    # Encoding layers
    layer = BatchNormalization()(model_2)
    encoding_1 = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))(layer, layer)
    encoding_1 = BatchNormalization()(encoding_1)
    encoding_1 = Activation(activation='relu')(encoding_1)
    encoding_1 = Dense(64)(encoding_1)

    layer1 = BatchNormalization()(encoding_1)
    encoding_2 = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))(layer1, layer1)
    encoding_2 = BatchNormalization()(encoding_2)
    encoding_2 = Activation(activation='relu')(encoding_2)
    encoding_2 = Dense(16)(encoding_2)

    layer2 = BatchNormalization()(encoding_2)
    encoding_3 = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))(layer2, layer2)
    encoding_3 = BatchNormalization()(encoding_3)
    encoding_3 = Flatten()(encoding_3)

    # Add Relational Transformer
    expand_dims = ExpandDimsLayer(axis=1)(encoding_3)
    relational_transformer = RelationalTransformer(embed_dim=16, num_heads=2, ff_dim=32, num_layers=2)
    relational_output = relational_transformer(expand_dims)

    relational_output = Flatten()(relational_output)
    relational_output = Activation(activation='relu')(relational_output)

    # Final classification layers
    classification_output = Dense(16, activation="softmax")(relational_output)
    classification_output = Dense(6)(classification_output)

    # Build SSA_TNet model with relational transformer
    SSA_TNet_model = Model(input_1, classification_output, name="SSA_TNet_with_RT")
    return SSA_TNet_Classifi()


def SSARTNet_Model(X, Y):
   
    # Initialize and compile model
    SSA_RTNet_model = SSA_TNet1_with_RelationalTransformer(input_shape=(250, 250, 1))
    weight = Adam(lr=Opt_COA.optimizeweight(), decay=1e-6)
    SSA_RTNet_model.compile(optimizer=weight, loss="categorical_crossentropy",metrics=['accuracy'])
    history = SSA_RTNet_model.fit(X,Y,epochs=1,batch_size=32,verbose=False)
    SSA_RTNet_model.summary()
    # SSA_TNet_model.save("model_saved/SSARTNet")
    SSA_TNet_history=history.history['accuracy']
    model = assarray(SSA_TNet_history)
    print()
    print("Accuracy:", model)
    print()


# # Call the function
# SSARTNet_Model(1, 1)
