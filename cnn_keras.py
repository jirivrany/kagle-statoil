
# coding: utf-8



import pandas as pd 
import numpy as np 
import cv2 # Used to manipulated the images 
np.random.seed(1207) # The seed I used - pick your own or comment out for a random seed. A constant seed allows for better comparisons though

# Import Keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.morphology import reconstruction


import model_simple as model_source


# ## Load Training Data

df_train = pd.read_json('./input/train.json') # this is a dataframe


# Isolation function.
def iso(arr):
    image = img_as_float(np.reshape(np.array(arr), [75,75]))
    image = gaussian_filter(image,2.5)
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image 
    dilated = reconstruction(seed, mask, method='dilation')
    return image-dilated



# Need to reshape and feature scale the images:

def get_scaled_imgs_4bands(df):
    imgs = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        iso1 = iso(row['band_1'])
        iso2 = iso(row['band_2'])
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        
        imgs.append(np.dstack((a, b, iso1, iso2)))

    return np.array(imgs)


def get_scaled_imgs(df):
    imgs = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)    


Xtrain = get_scaled_imgs(df_train)


# Get the response variable "is_iceberg"

Ytrain = np.array(df_train['is_iceberg'])


# Some of the incident angle from the satellite are unknown and marked as "na". Replace these na with 0 and find the indices where the incident angle is >0 (this way you can use a truncated set or the full set of training data).

df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)

# You can now use the option of training with only known incident angles or the whole set. I found slightly better results training with only the known incident angles so:

Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]

# ## Adding images for training

# Now, the biggest improvement I had was by adding more data to train on. I did this by simply including horizontally and vertically flipped data. Using OpenCV this is easily done.

def get_more_images(imgs):
    
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
      
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
        
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
        
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
      
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
       
    more_images = np.concatenate((imgs,v,h))
    
    return more_images


# I rename the returned value so i have the option of using the original data set or the expanded data set

Xtr_more = get_more_images(Xtrain) 

# And then define the new response variable:

Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))


# ## CNN Keras Model


model = model_source.get_model()
model.summary()

MODEL_FILE = '.mdl_simple_dr_w.hdf5'
SUBMISSION = 'sub_simple_drop.csv'
batch_size = 32
earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
mcp_save = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, verbose=1, epsilon=1e-4, mode='min')

# Now train the model! (Each epoch ran at about 10s on GPU)


model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=40, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)
#model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=40, verbose=1, callbacks=[ mcp_save, reduce_lr_loss], validation_split=0.1)

# ## Results

# Load the best weights and check the score on the training data.


model.load_weights(filepath = MODEL_FILE)

score = model.evaluate(Xtrain, Ytrain, verbose=1)
print('Train score:', score[0])
print('Train accuracy:', score[1])

# Now, to make a submission, load the test data and train the model and output a csv file.

df_test = pd.read_json('./input/test.json')
df_test.inc_angle = df_test.inc_angle.replace('na',0)
Xtest = (get_scaled_imgs(df_test))
pred_test = model.predict(Xtest)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
print(submission.head(10))

submission.to_csv(SUBMISSION, index=False)

