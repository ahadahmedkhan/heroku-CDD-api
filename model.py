
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import numpy as np
import keras.backend as B
from keras.preprocessing.image import ImageDataGenerator
import pickle

image_width = 50
image_height = 50

# Load Datasets

train_data_dir='Datasets/train'
test_data_dir='Datasets/Test'

# preprocess data

epochs=20
batch_size=10

if B.image_data_format()=='channel first':
    input_shape = (3,image_height,image_width)
else:
    input_shape=(image_height,image_width,3)

train_datagen=ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,

)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_height,image_width),
    batch_size=batch_size,
    # color_mode='grayscale',
    class_mode='categorical',
    )
# define model architecture

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(16,(3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(7))
model.add(Activation('softmax'))

 # Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
# fit the model on train dataet

model.fit_generator(

    train_generator,
    steps_per_epoch=29,
    epochs= epochs
)

model.save('model.h5')

# serialization
# pickle_out = open('model.pkl','wb')
# pickle._dump(model,pickle_out)
# pickle_out.close()

# deserialization
# pickle_in = open('model.pkl','rb')
# result = pickle._load(pickle_in)
#
# img=image.load_img('Datasets\\Test\\rust\\9.jpg',target_size=(image_height,image_width))
# img=image.img_to_array(img=img)
# img=np.expand_dims(img,axis=0)
#
#
# result=model.predict(img)
# print(result)

