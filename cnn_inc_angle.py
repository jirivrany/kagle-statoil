import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input,  Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import cv2
import keras

np.random.seed(1207)


"""

"""


def get_scaled_imgs(df):
    imgs = []

    for i, row in df.iterrows():
        # make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2  # plus since log(x*y) = log(x) + log(y)

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

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs, v, h))

    return more_images


def getModel():
    # Build keras model

    image_model = Sequential()

    # CNN 1
    image_model.add(Conv2D(64, kernel_size=(3, 3),
                           activation='relu', input_shape=(75, 75, 3)))
    image_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    image_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    image_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    # CNN 2
    image_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    image_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    image_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # CNN 3
    image_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    image_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    image_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # CNN 4
    image_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # You must flatten the data for the dense layers
    image_model.add(Flatten())

    # Image input encoding
    image_input = Input(shape=(75, 75, 3))
    encoded_image = image_model(image_input)

    # Inc angle input
    inc_angle_input = Input(shape=(1,))

    # Combine image and inc angle
    combined = keras.layers.concatenate([encoded_image, inc_angle_input])

    dense_model = Sequential()

    # Dense 1
    dense_model.add(Dense(512, activation='relu', input_shape=(257,)))
    dense_model.add(Dropout(0.2))

    # Dense 3
    dense_model.add(Dense(256, activation='relu'))
    dense_model.add(Dropout(0.2))

    # Output
    dense_model.add(Dense(1, activation="sigmoid"))

    output = dense_model(combined)

    # Final model
    combined_model = Model(
        inputs=[image_input, inc_angle_input], outputs=output)

    optimizer = Adam(lr=0.0001, decay=0.0)
    combined_model.compile(loss='binary_crossentropy',
                           optimizer=optimizer, metrics=['accuracy'])

    return combined_model

df_train = pd.read_json('./input/train.json')
Xtrain = get_scaled_imgs(df_train)
Ytrain = np.array(df_train['is_iceberg'])

df_train.inc_angle = df_train.inc_angle.replace('na', 0)
idx_tr = np.where(df_train.inc_angle > 0)

Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0], ...]
Xinc = df_train.inc_angle[idx_tr[0]]

#Xtrain = get_more_images(Xtrain)
#Xinc = np.concatenate((Xinc, Xinc, Xinc))
#Ytrain = np.concatenate((Ytrain, Ytrain, Ytrain))

model = getModel()
model.summary()

model_file = '.mdl_angle2_wts.hdf5'
batch_size = 32
earlyStopping = EarlyStopping(
    monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint(
    model_file, save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=8, verbose=1, epsilon=1e-4, mode='min')

model.fit([Xtrain, Xinc], Ytrain, batch_size=batch_size, epochs=50, verbose=1,
          callbacks=[mcp_save, reduce_lr_loss], validation_split=0.2)


model.load_weights(filepath=model_file)
score = model.evaluate([Xtrain, Xinc], Ytrain, verbose=1)
print('Train score:', score[0])
print('Train accuracy:', score[1])

df_test = pd.read_json('./input/test.json')
df_test.inc_angle = df_test.inc_angle.replace('na', 0)
Xtest = (get_scaled_imgs(df_test))
Xinc = df_test.inc_angle
pred_test = model.predict([Xtest, Xinc])

submission = pd.DataFrame(
    {'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
print(submission.head(10))

submission.to_csv('sub_inc_angle_noaug.csv', index=False)
