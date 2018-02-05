    
# coding: utf-8



import pandas as pd 
import numpy as np 
import cv2 # Used to manipulated the images 
seed = 1207
np.random.seed(seed) 

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

from sklearn.model_selection import StratifiedKFold

import model_simple_nodrop as model_source


# ## Load Training Data

df_train = pd.read_json('./input/train.json') # this is a dataframe



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


Xtrain = get_scaled_imgs(df_train)
Ytrain = np.array(df_train['is_iceberg'])
df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)

Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]



Xtr_more = get_more_images(Xtrain) 
Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
for fold_n, (train, test) in enumerate(kfold.split(Xtr_more, Ytr_more)):
    print("FOLD nr: ", fold_n)
    model = model_source.get_model()
    #model.summary()

    MODEL_FILE = 'mdl_simple_k{}_wght.hdf5'.format(fold_n)
    batch_size = 32
    mcp_save = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, verbose=1, epsilon=1e-4, mode='min')


    model.fit(Xtr_more[train], Ytr_more[train],
        batch_size=batch_size,
        epochs=32,
        verbose=1,
        validation_data=(Xtr_more[test], Ytr_more[test]),
        callbacks=[mcp_save, reduce_lr_loss])
    
    model.load_weights(filepath = MODEL_FILE)

    score = model.evaluate(Xtr_more[test], Ytr_more[test], verbose=1)
    print('\n Val score:', score[0])
    print('\n Val accuracy:', score[1])

    SUBMISSION = './result/simplenet/sub_simple_v1_{}.csv'.format(fold_n)

    df_test = pd.read_json('./input/test.json')
    df_test.inc_angle = df_test.inc_angle.replace('na',0)
    Xtest = (get_scaled_imgs(df_test))
    pred_test = model.predict(Xtest)

    submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
    print(submission.head(10))

    submission.to_csv(SUBMISSION, index=False)
    print("submission saved")


# Stack all
wdir = './result/simplenet/'
stacked_1 = pd.read_csv(wdir + 'sub_simple_v1_0.csv')
stacked_2 = pd.read_csv(wdir + 'sub_simple_v1_1.csv')
stacked_3 = pd.read_csv(wdir + 'sub_simple_v1_2.csv')
stacked_4 = pd.read_csv(wdir + 'sub_simple_v1_3.csv')
stacked_5 = pd.read_csv(wdir + 'sub_simple_v1_4.csv')
stacked_6 = pd.read_csv(wdir + 'sub_simple_v1_5.csv')
stacked_7 = pd.read_csv(wdir + 'sub_simple_v1_6.csv')
stacked_8 = pd.read_csv(wdir + 'sub_simple_v1_7.csv')
stacked_9 = pd.read_csv(wdir + 'sub_simple_v1_8.csv')
stacked_10 = pd.read_csv(wdir + 'sub_simple_v1_9.csv')
sub = pd.DataFrame()
sub['id'] = stacked_1['id']
sub['is_iceberg'] = np.exp(np.mean(
    [
        stacked_1['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_2['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_3['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_4['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_5['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_6['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_7['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_8['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_9['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_10['is_iceberg'].apply(lambda x: np.log(x)),
        ], axis=0))

sub.to_csv(wdir + 'final_ensemble.csv', index=False, float_format='%.6f')    

