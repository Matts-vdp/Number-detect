import numpy as np
import mnist #dataset
import time
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

ep = int(input('epochs: '))

train_images = mnist.train_images()
train_labels = mnist.train_labels()


# Normalize the images.
train_images = (train_images / 255) - 0.5


# Flatten the images.
train_images = train_images.reshape((train_images.shape[0],28,28,1))






name = 'convtrain-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(name))

#wip
model = Sequential([
    Conv2D(32, (3,3),activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax'),
])

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'],
)

print('started training')
#train
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs = ep,
    batch_size = 32,
    validation_split=0.1,
    callbacks=[tensorboard]
)

model.save_weights('num.model')





input('Done....')
