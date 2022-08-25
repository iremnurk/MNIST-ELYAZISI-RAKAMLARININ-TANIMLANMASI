# -*- coding: utf-8 -*-
"""
Created on Fri May  3 01:38:06 2019

@author: Suheda
"""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')


train = pd.read_csv("train.csv")
print(train.shape)
train.head()


test= pd.read_csv("test.csv")
print(test.shape)
test.head()

Y_train = train["label"]
# X_traindeki birinci sütun rakamların etiketini içerir.
X_train = train.drop(labels = ["label"],axis = 1) 

# Verilerin normalize edilmesi
X_train = X_train / 255.0
test = test / 255.0
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)

# Yeniden şekillendirilmesi
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)

# Etiket kodlaması
from keras.utils.np_utils import to_categorical # etiketlerin one-hot-encoding yapılması
Y_train = to_categorical(Y_train, num_classes = 10)

# Split ile eğitim verilerinin 0.10nu modelde kullanılacak test verisi haline getirilir.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
print("x_train shape",X_train.shape)
print("x_test shape",X_test.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)



from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
# 5x5 boyutunda filtereler oluşturulur ve modele eklenir.
#Modelimize bir 2 boyutlu Convolutional katman(layer) ekler. İlk parametresi kaç adet filtrenin bu katmanda kullanılacağıdır. İkinci parametre filtrenin/kernelin boyutudur. 
#Bu modelin ilk katmanı olduğu için input_shape parametresi vermemiz gereklidir.
# "Padding" fotoğrafa çervçeve ekler ve çıkış boyutunun giriş boyutuna eşit olması sağlanır.
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
#MaxPooling işlemi, verimizden, verilen pool_size boyutunda kümeler alıp bu kümeler içerisindeki en büyük değerleri kullanarak yeni bir matris oluşturur.
#Oluşan matrisin boyutu daha küçüldüğü için sonraki katmanlarda işlem hızımızı arttıracaktır ayrıca MaxPooling overfit durumunun önüne geçer.
model.add(MaxPool2D(pool_size=(2,2)))
# Rastgele olacak şekilde nöronların %25'ini kapatıyoruz: (Eğitim sırasındaki ezberlemeyi önlemek için.)
model.add(Dropout(0.25))

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))



#Flatten metodu çok boyutlu olan verimizi tek boyutlu hale getirerek standart yapay sinir ağı için hazır hale getirir.
model.add(Flatten())
#Bir standart yapay sinir ağı katmanı oluşturur
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
model.summary()

# Optimizer tanulanır
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# Model derlnir
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 40 
batch_size = 250
# veri arttırma
datagen = ImageDataGenerator(
        featurewise_center=False,  #veri kümesi üzerinden giriş ortalamasını 0 olarak ayarlanır
        samplewise_center=False,  # her örnek ortalamasını 0 olarak ayarlayın
        featurewise_std_normalization=False,  #girişleri veri kümesinin std. normalizasyonu ile böl 
        samplewise_std_normalization=False,  # her girişi std'ye bölün
        zca_whitening=False,  # boyut küçültme
        rotation_range=0.5,  # 5 derece aralığındaki görüntüleri rastgele döndür
        zoom_range = 0.5, # rastgele zoom görüntüsü 5%
        width_shift_range=0.5,  #rastgele görüntüleri yatay olarak%5 kaydır
        height_shift_range=0.5,  # rastgele görüntüleri dikey olarak% 5 kaydır
        horizontal_flip=False,  # rastgele görüntüleri çevirin
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)


history = model.fit_generator(datagen.flow(X_train,Y_train, 
                              batch_size=batch_size),
                              epochs = epochs,
                              verbose=1,
                              validation_data = (X_test,Y_val), 
                              steps_per_epoch=X_train.shape[0] // batch_size)
model.save("model.h5")

score = model.evaluate(X_test,Y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Kayıp Eğrilerini çizdirelim
plt.figure(figsize=[10,5])
plt.subplot(121)
plt.plot(history.history['loss'], 'ro', label='Training loss')
plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
plt.legend()
plt.xlabel('Epochs ')
plt.ylabel('Loss')
plt.title('Loss Curves')

plt.show()
