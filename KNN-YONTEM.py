# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:44:17 2019

@author: iremn
"""

#gerekli paketler import edilir
from __future__ import print_function
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.color import rgb2gray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import imutils
import numpy as np
#import -- opencv ve matplot kutuphaneleri import edilir
import cv2
import matplotlib.pyplot as plt
 
#MNIST rakam veri seti yüklenir
mnist = datasets.load_digits()


#Veririlerimizi   eğitim ve test bölümlerini oluşturmamıza yardımcı olmak için,
#scikit-learn tarafından sağlanan  train_test_split fonk.ile  içe  aktarırız .
#MNIST verilerini alınır eğitim ve test verileri olmak üzere ikiye ayrılır
# ve bu MNIST verisinin % 75'i  eğitim % 25'i test için kullanılcaktır.
(egitimVerisi, testVerisi, egitimEtiketleri, testEtiketleri) = train_test_split(np.array(mnist.data),
mnist.target, test_size=0.25, random_state=42)

 


#Bir doğrulama setine de ihtiyacımız var, böylece
# k'nin değerini ayarlayabilelim .Eğitim verilerimizi bölümlere ayırarak
#doğrulama setimizi oluşturuyoruz  - eğitim verilerinin% 10'u validasyona, geri 
#kalan% 90'ı ise eğitim verileri olarak kalacak.

#Kısıca, eğitim verilerinin% 10'unu alıp , bunları doğrulama için kullandım.
(egitimVerisi, valVerisi, egitimEtiketleri, valEtiketleri) = train_test_split(egitimVerisi, egitimEtiketleri,
test_size=0.1, random_state=84)

 
#her veri bölmesinin boyutunu göster
#train validation test için
print("Eğitim verisi: {}".format(len(egitimEtiketleri)))
print("Doğrulama verisi: {}".format(len(valEtiketleri)))
print("Test verisi: {}".format(len(testEtiketleri)))




kDegerleri = range(1, 30, 2)# k-En Yakın Komşu sınıflandırıcı için k değerlerini,tanımlıyoruz
dogruluklar = [] # k değerinin doğruluk listesi

# k-En Yakın Komşu sınıflandırıcı için `k` değerlerinin üzerinde döngü olustururuz

for k in range(1, 30, 2):
         
          #bu k değerlerinin her birine döngü uyguladık
          #KNeighborsClassifier'ı eğitiyoruz, egitim verilerini
          #ve eğitim etiketlerini modelin uygun yöntemine ulaştırıyoruz(model.fit)
          model = KNeighborsClassifier(n_neighbors=k)
          model.fit(egitimVerisi, egitimEtiketleri)
        
          #modeli değerlendirip  , doğruluk listesini güncelledik
          skor = model.score(valVerisi, valEtiketleri)
          print("k=%d, accuracy=%.2f%%" % (k, skor * 100))
          dogruluklar.append(skor)
          
# en büyük kesinliğe sahip olan k değerini bulunmus olur
i = np.argmax(dogruluklar)
print("k=%d ile elde edilen en yüksek doğruluk %.2f%% " % (kDegerleri[i],
dogruluklar[i] * 100))
#ÖZETLE bu kısımda 
#Modelimiz eğitildikten sonra, doğrulama verilerimi
# kullanılarak değerlendirmemiz gerekir
#Modelimizin puanlama yöntemi; k-NN sınıflandırıcımızın kaç tahminin 
#doğru olduguna bakar .
#Daha sonra bu puanı alırız ve doğruluk listemizi güncelleriz,
# böylece validasyon setinde en yüksek doğruluğu elde eden k değerini belirleyebiliriz


# Sınıflandırıcımızı en iyi k değerini kullanarak yeniden eğitiriz
# sadece  en yüksek doğruluğu elde eden k'nin değerini  alıyoruz, KNeighborsClassifier'ı
# bu k değerini kullanarak  yeniden  eğitiyoruz ve ardından  
#çıktısını görebilmek adına, classification_report fonk. kullanarak performansı değerlendiriyoruz.

model = KNeighborsClassifier(n_neighbors=kDegerleri[i])
model.fit(egitimVerisi, egitimEtiketleri)
tahmin_et = model.predict(testVerisi)

print("TEST VERİLERİ DEĞERLENDİRMESİ")
print(classification_report(testEtiketleri, tahmin_et))
#classification_report işlevi, sınıflandırıcımızın performansını 
#değerlendirmemize yardımcı olacak kullanışlı bir küçük araçtır.

print ("Karışıklık matrisi")
print(confusion_matrix(testEtiketleri,tahmin_et))


#test setimizden  rastgele görüntü üzerinde döngü yapıyoruz.
for i in np.random.randint(0, high=len(testEtiketleri), size=(1,)):
    
         # görüntüyü tut ve sınıflandır
         resim = testVerisi[i]
         tahmin = model.predict([resim])[0] 
         #rastgele görüntüyü alır ve görüntünün hangi rakamı içerdiğini tahmin eder      
         imgdata = np.array(resim, dtype='float')
         pixels = imgdata.reshape((8,8))
         plt.imshow(pixels,cmap='gray')
         plt.annotate(tahmin,(3,3),bbox={'facecolor':'white'},fontsize=16)
         # tahmini göster
         print("Rakam bu olmalı: {}".format(tahmin))
         plt.show()
         cv2.waitKey(150)
         

