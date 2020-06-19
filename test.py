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

# 測試單一資料並印出該圖片
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
    
# 測試folder內所有資料    
train_datagen = ImageDataGenerator(rescale = 1.0 / 255 )
test_datagen = ImageDataGenerator(rescale = 1.0 / 255 )

train_generator = train_datagen.flow_from_directory(
        'data3/train',
        target_size = (224, 224),
        class_mode = 'categorical')
test_generator = test_datagen.flow_from_directory('Test_data', target_size=(224, 224),batch_size=1,class_mode='categorical', shuffle=False,)

test_generator.reset()
pred = model.predict_generator(test_generator)

predicted_class_indices = np.argmax(pred, axis=1)
labels = (train_generator.class_indices)
label = dict((v,k) for k,v in labels.items())

# 建立test標籤與真實標籤的關係
predictions = [label[i] for i in predicted_class_indices]

#建立預測結果和文件名之間的關係
filenames = test_generator.filenames
for idx in range(len(filenames )):
    print('predict  %s' % ((predictions[idx])))
    print('title    %s' % filenames[idx])
    print('')
