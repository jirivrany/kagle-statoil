
# coding: utf-8



import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

np.random.seed(1207) # The seed I used - pick your own or comment out for a random seed. A constant seed allows for better comparisons though

# Import Keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import model_simple as model_source

MODEL_FILE = '.mdl_kerasgen_simple_w.hdf5'
SUBMISSION = 'simple_32batch.csv'


# ## Load Training Data

df_train = pd.read_json('./input/train.json') # this is a dataframe


# Need to reshape and feature scale the images:

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


Xdata = get_scaled_imgs(df_train)


# Get the response variable "is_iceberg"

Ydata = np.array(df_train['is_iceberg'])


# Some of the incident angle from the satellite are unknown and marked as "na". Replace these na with 0 and find the indices where the incident angle is >0 (this way you can use a truncated set or the full set of training data).

df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)

# You can now use the option of training with only known incident angles or the whole set. I found slightly better results training with only the known incident angles so:

Ydata = Ydata[idx_tr[0]]
Xdata = Xdata[idx_tr[0],...]



model = model_source.get_model()
model.summary()
X_train, X_val, Y_train, Y_val = train_test_split(Xdata, Ydata, test_size = 0.15, random_state=1207)

#batch_size = 32
mcp_save = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1, epsilon=1e-4, mode='min')
#
#
#
##model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)
#model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=60, verbose=1, callbacks=[mcp_save, reduce_lr_loss])
#
# ## Results


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=7, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


epochs = 30 
batch_size = 32

datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images


datagen.fit(X_train)

# Fit the model
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                epochs = epochs,
                                validation_data = (X_val,Y_val),
                                verbose = 1,
                                steps_per_epoch=X_train.shape[0] * 10 // batch_size,
                                callbacks=[learning_rate_reduction, mcp_save,  reduce_lr_loss])




model.load_weights(filepath = MODEL_FILE)

score = model.evaluate(Xdata, Ydata, verbose=1)
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

