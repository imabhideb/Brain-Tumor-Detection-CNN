import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10EpochsCategorical.h5')

image = cv2.imread('D:\\Final Project -Python\\pred\\pred10.jpg')
img = Image.fromarray(image)

img = img.resize((64,64))

img = np.array(img)

input_img = np.expand_dims(img, axis=0)

result = np.argmax(model.predict(input_img), axis=-1)     # use this because the latest tensorflow doesn't use predict_class anymore
# result = model.predict(input_img)    # predict_classes(input_img)
print(result)