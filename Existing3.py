import numpy as np 
from tensorflow import keras
import tensorflow as tf
from keras.layers import Input, Dense, InputLayer, Conv2D,Dropout, MaxPooling2D, Flatten, Reshape,LSTM, Bidirectional
from tensorflow.keras import layers, Model
from dbn.tensorflow import SupervisedDBNClassification as DBNClassification

# x,y,x1,y1
def existing_classifiers(x,Y):
    
    # SNN
    print("-------------------SNN-----------------")
    input_layer = Input((250,250,1))
    layer1 = Conv2D(16, (3,3), activation='relu', padding='same')(input_layer)
    layer2 = MaxPooling2D((2,2), padding='same')(layer1)
    layer3 = Conv2D(16, (3,3), activation='relu', padding='same')(layer2)
    layer4 = MaxPooling2D((2,2), padding='same')(layer3)
    layer5 = Flatten()(layer4)
    xx1 = layers.Dense(64, activation="relu", name="dense_1")(layer5)
    xx2 = layers.Dense(64, activation="relu", name="dense_2")(xx1)
    outputs = layers.Dense(1, name="softmax")(xx2)
    SNN_model = Model(inputs=input_layer, outputs=outputs)  
    SNN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    SNN_model.fit(x,Y,epochs=300,verbose=True)
    SNN_model.save("Exis_model/SNN")
    # SNN_model.summary()
    
 
    # SAE
    print("-------------------SAE-----------------")
    input_layer = Input((250,250, 1))
    encoder = Dense (64, activation='relu')(input_layer)
    encoder = Dense (64, activation='relu')(encoder)
    encoder = Dense (32, activation='relu')(encoder)
    encoder = Dense (32, activation='relu')(encoder)
    encoder = Flatten()(encoder)
    decoder = Dense(16, activation='relu')(encoder)
    decoder = Dense(64, activation='relu')(decoder)
    decoder = Dense(1, activation='softmax')(decoder)
    SAE_model = Model(inputs=input_layer, outputs=decoder)
    SAE_model.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.SGD(lr=.1))
    SAE_model.fit(x,Y,epochs=300,verbose=True)
    SAE_model.save("Exis_model/SAE")
    # SAE_model.summary()


    # BILSTM
    print("-------------------BILSTM-----------------")
    input_layer = Input((250,250,1))
    model_3 = Reshape((-1, 1))(input_layer)
    layer2 = Bidirectional(LSTM(64))(model_3)
    layer3 = Dropout(0.5)(layer2)
    layer4 = Dense(32, activation='relu')(layer3)
    outputs = layers.Dense((1), name="softmax")(layer4)
    bilstm_model = Model(inputs=input_layer, outputs=outputs)
    bilstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    bilstm_model.fit(x,Y,epochs=300,verbose=True)
    bilstm_model.save("Exis_model/BILSTM")
    # bilstm_model.summary()
    
   
    # CNN
    print("-------------------CNN-----------------")
    input_layer = Input((250,250, 1))
    layer1 = Conv2D(16, (3,3), padding="same",activation='relu')(input_layer)
    layer2 = MaxPooling2D((2,2))(layer1)
    layer3 = Conv2D(16, (3,3), activation='relu', padding="same")(layer2)
    layer5 = Conv2D(16, (3,3), activation='relu', padding="same")(layer3)
    layer6 = Flatten()(layer5)
    layer7 = Dense(16, activation='relu')(layer6)
    layer8 = Dense(32)(layer7)
    outputs = layers.Dense(1, activation="relu", name="softmax")(layer8)
    CNN_model = Model(inputs=input_layer, outputs=outputs)  
    CNN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    CNN_model.fit(x,Y,epochs=300,verbose=True)
    CNN_model.save("Exis_model/CNN")
    # CNN_model.summary()
    
    # DBN
    print("-------------------DBN-----------------")
    dbn_classifier = DBNClassification(hidden_layers_structure=[32, 250, 250],batch_size=32,dropout_p=0.2,
                                              learning_rate=0.1, n_epochs_rbm=1,activation_function='relu', verbose=False)
    dbn_classifier.fit(x,Y)
    dbn_classifier.save("Exis_model/DBN")
  
    return SNN_model, SAE_model, bilstm_model, CNN_model
    
