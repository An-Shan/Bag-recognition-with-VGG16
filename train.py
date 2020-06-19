from keras import applications
VGG16 = applications.VGG16()
VGG16.layers.pop()

from keras import Sequential
model = Sequential() 
for layer in VGG16.layers:
    model.add(layer)

# only train parameters from the last conv layers to the softmax layer 
for layer in model.layers[:-8]:
  layer.trainable = False

from keras.layers import Dense
model.add(Dense(11, activation = 'softmax'))

from keras.optimizers import Adam
# hyperparameter can be tune 
opt = Adam(learning_rate = 0.0012 )
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

#data preprocess
from keras.preprocessing.image import  ImageDataGenerator,image
datagen = ImageDataGenerator(rescale = 1.0 / 255, brightness_range = [0.8,1.2],  zoom_range = [0.8,1.2], width_shift_range = 0.3, height_shift_range = 0.3, horizontal_flip = True)

from PIL import Image,ImageFile
train_generator = datagen.flow_from_directory(
        'data3/train',
        target_size = (224, 224),
        batch_size = 64,
        classes = ['BackPack 雙肩後背包','BucketBag 水桶包','DuffleBag 旅行包','Fannypack 隨身腰包','Luggage 旅行箱','MessengerBag 郵差包','SaddleBag 馬鞍包','Satchel 劍橋包','Sleepingbag 睡袋','Tote 托特包','else 其他'],
        class_mode = 'categorical')
print(train_generator.class_indices)

validation_generator = datagen.flow_from_directory(
        'data3/valid',
        target_size = (224, 224),
        batch_size = 64,
        classes = ['BackPack 雙肩後背包','BucketBag 水桶包','DuffleBag 旅行包','Fannypack 隨身腰包','Luggage 旅行箱','MessengerBag 郵差包','SaddleBag 馬鞍包','Satchel 劍橋包','Sleepingbag 睡袋','Tote 托特包','else 其他'],
        class_mode = 'categorical')
print(validation_generator.class_indices)

# start training
model.fit_generator(
        train_generator,
        steps_per_epoch = 95,
        epochs = 100,
        validation_data = validation_generator,
        validation_steps = 14)

#plot loss and accuracy history
import matplotlib.pyplot as plt
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

model.save_weights('/content/drive/My Drive/0619_01_weights.h5')
