
# coding: utf-8

"""

"""


import pandas as pd 
import numpy as np 
import cv2 # Used to manipulated the images 
from scipy.signal import wiener

np.random.seed(1207) # The seed I used - pick your own or comment out for a random seed. A constant seed allows for better comparisons though

# Import Keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# ## Load Training Data

# In[2]:


df_train = pd.read_json('./input/train.json') # this is a dataframe


# Need to reshape and feature scale the images:

# In[3]:


def get_scaled_imgs(df):
    imgs = []
    
    for i, row in df.iterrows():
        band_1 = np.array(row['band_1'])
        band_2 = np.array(row['band_2'])

        #make 75x75 image
        band_1 = band_1.reshape(75, 75)
        band_2 = band_2.reshape(75, 75)
        #band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        #c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b)))

    return np.array(imgs)




def get_more_images(imgs):
    
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
      
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        #c=imgs[i,:,:,2]
        
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        #cv=cv2.flip(c,1)
        #ch=cv2.flip(c,0)
        
        #vert_flip_imgs.append(np.dstack((av, bv, cv)))
        #hori_flip_imgs.append(np.dstack((ah, bh, ch)))
        vert_flip_imgs.append(np.dstack((av, bv)))
        hori_flip_imgs.append(np.dstack((ah, bh)))
      
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
       
    more_images = np.concatenate((imgs,v,h))
    
    return more_images


def getModel():
    #Build keras model
    
    model=Sequential()
    
    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Dropout(0.2))

    #CNN 4
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # You must flatten the data for the dense layers
    model.add(Flatten())

    #Dense 1
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    #Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.0001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model



Xtrain = get_scaled_imgs(df_train)
Ytrain = np.array(df_train['is_iceberg'])
df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)

Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]

#Xtr_more = get_more_images(Xtrain) 
#Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))

X_train, X_valid, y_train, y_valid = train_test_split(Xtrain, Ytrain, test_size=0.1)

X_train_more = get_more_images(X_train)
y_train_more = np.concatenate([y_train, y_train, y_train])
X_valid_more = get_more_images(X_valid)
y_valid_more = np.concatenate([y_valid, y_valid, y_valid])


model = getModel()
model.summary()

batch_size = 32
model_file = '.mdl_2l2_wts.hdf5'

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint(model_file, save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-6, mode='min')


#model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)
#model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=60, verbose=1, callbacks=[mcp_save, reduce_lr_loss], validation_split=0.2)

model.fit(X_train_more, y_train_more, batch_size=32, epochs=60, verbose=1,
                     callbacks=[mcp_save, reduce_lr_loss],
                     validation_data=(X_valid, y_valid))


model.load_weights(filepath = model_file)

score = model.evaluate(Xtrain, Ytrain, verbose=1)
print('Train score:', score[0])
print('Train accuracy:', score[1])


df_test = pd.read_json('./input/test.json')
df_test.inc_angle = df_test.inc_angle.replace('na',0)
Xtest = (get_scaled_imgs(df_test))
pred_test = model.predict(Xtest)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
print(submission.head(10))

submission.to_csv('sub-2bands-nodrop-aug.csv', index=False)

