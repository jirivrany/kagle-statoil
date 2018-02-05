# Import Keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


def get_model():
    
    """
    Epoch 50/50
    3530/3530 [==============================] - 10s - loss: 8.5420e-04 - acc: 1.0000 - val_loss: 0.3877 - val_acc: 0.9083
    1471/1471 [==============================] - 1s     
    Train score: 0.00226768349974
    Train accuracy: 1.0

    """
    
    model=Sequential()
    
    # Block 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation=None, input_shape=(75, 75, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(3, 3),activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))     
    
    # Block 2
    model.add(Conv2D(128, kernel_size=(3, 3),activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=(3, 3),activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=(3, 3),activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))     
    
    # Block 3

    model.add(Conv2D(256, kernel_size=(3, 3),activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(3, 3),activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(3, 3),activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(3, 3),activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(3, 3),activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(3, 3),activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   
    

    # You must flatten the data for the dense layers
    model.add(Flatten())

    #Dense 1
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    #Dense 2
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    #Dense 3
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

     #Dense 3
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.0001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model
