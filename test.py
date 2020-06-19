from keras.models import load_model
VGG16 = load_model('VGG16.h5')
VGG16.layers.pop()
from keras import Sequential
model = Sequential()
for layer in VGG16.layers:
    model.add(layer)

from keras.layers import Dense
model.add(Dense(11, activation = 'softmax'))

model.load_weights('0619_01_weights.h5')

from keras.preprocessing.image import  image
import numpy as np
import matplotlib.pyplot as plt

try:
  img = image.load_img("/content/drive/My Drive/Test_data/07.jpg",target_size=(224,224))
  img = np.asarray(img)
  img = np.expand_dims(img, axis=0)

  plt.imshow(img[0])
  output = list(model.predict(img)[0])
  print(output)
  product_list = ['BackPack 雙肩後背包','BucketBag 水桶包','DuffleBag 旅行包',
  'FannyPack 隨身腰包','Luggage 旅行箱','MessengerBag 郵差包',
  'SaddleBag 馬鞍包','Satchel 劍橋包','Sleepingbag 睡袋','Tote 托特包','else 其他']
      
  # maxindex = output.index(max(output))
  # print(maxindex)
  print({'msg' : '您是否是要查詢:' + str(product_list[np.argmax(output)])})

except Exception as e:
    print({'msg':str(e).split(':')[0]})