from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import cv2 as cv



# Build the model.
model = Sequential([
    Conv2D(32, (3,3),activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax'),
])

# Load the model's saved weights.
model.load_weights('num.model')

file = str(input('filename: '))

image = cv.imread(file, cv.IMREAD_GRAYSCALE)
image = cv.resize(image, (28,28))
image = 255-image        


plt.imshow(image.reshape(28, 28),cmap='Greys')
plt.show()

image = (image/255) - 0.5

pred = model.predict(image.reshape(1,28,28,1), batch_size=1)


p = pred * 10000

for i in range(len(p[0])):
    p[0][i] = int(p[0][i])
    p[0][i] /= 100.0
    print(i,' : ', p[0][i],'%')

print('result: ',pred.argmax())

print()

input('Done..')
