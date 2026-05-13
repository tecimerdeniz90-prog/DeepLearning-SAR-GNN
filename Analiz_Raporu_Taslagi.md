# DERİN ÖĞRENME PROJESİ ANALİZ RAPORU

**Öğrenci Adı Soyadı:** [Adınızı Soyadınızı Yazın]
**Öğrenci Numarası:** [Numaranızı Yazın]
**Ders:** Derin Öğrenme
**Konu:** Graf Sinir Ağları (GNN) ile SAR Görüntülerinde Hedef Sınıflandırma
**GitHub Linki:** https://github.com/tecimerdeniz90-prog/DeepLearning-SAR-GNN

---

## 1. GİRİŞ VE PROJE AMACI
Bu projenin amacı, Synthetic Aperture Radar (SAR) görüntüleri üzerindeki hedeflerin sınıflandırılması problemini, geleneksel Evrişimli Sinir Ağları (CNN) yerine modern bir yaklaşım olan Graf Sinir Ağları (GNN) kullanarak çözmektir. SAR görüntüleri, piksellerin doğrudan renk değerleri taşıdığı standart optik fotoğraflardan farklı olarak, yüzeyin radar dalgalarına verdiği yansıma (backscatter) bilgilerini içerir. Bu yapısal ve dokusal özellikleri yakalamak için görüntünün piksellerini bir ağ (graf) yapısına dönüştürerek GNN modellerinin gücünden faydalanılması hedeflenmiştir.

## 2. VERİ SETİ (MSTAR Veri Seti)
Projede MSTAR (Moving and Stationary Target Acquisition and Recognition) veri setinin yapısı baz alınmıştır. GNN mimarisinin bu veri üzerinde çalışabilmesi için görüntüler doğrudan piksel matrisleri olarak değil, parçalanmış ızgara (grid) tabanlı bir graf yapısına dönüştürülmüştür. 
- Her bir görüntü 8x8'lik yamalara (patch) bölünmüştür.
- Her bir yama grafın bir düğümünü (node) oluşturmuştur.
- Komşu yamalar (sağ, sol, alt, üst) birbirine kenarlar (edge) ile bağlanarak 4'lü komşuluk ilişkisi kurulmuştur.

## 3. MODEL MİMARİSİ: GRAF SİNİR AĞI (GCN)
Sınıflandırma problemi için **Graph Convolutional Network (GCN)** modeli tasarlanmıştır. PyTorch ve PyTorch Geometric kütüphaneleri kullanılarak oluşturulan model mimarisi şu şekildedir:
1. **GCN Katmanı 1:** Girdi düğüm özelliklerini alır (64 boyutlu) ve 64 gizli boyuta (hidden dimension) çevirir. ReLU aktivasyonu uygulanır.
2. **GCN Katmanı 2:** 64 boyutu 128 boyuta çıkarır. Düğümler komşularından daha geniş çaplı bilgi toplar.
3. **GCN Katmanı 3:** 128 boyutu tekrar 64 boyuta indirger.
4. **Global Pooling (Ortak Havuzlama):** Graf üzerindeki tüm düğümlerin ortalaması alınarak (global_mean_pool), grafın tamamını temsil eden tek bir vektör elde edilir.
5. **Tam Bağlı Katman (Fully Connected):** Havuzlamadan çıkan vektör, Dropout (aşırı öğrenmeyi/overfitting önlemek için) işleminden sonra sınıf sayısına göre çıktı üreten doğrusal (linear) bir katmana sokulur.

## 4. EĞİTİM (TRAINING) VE TEST SÜRECİ
Model, Adam optimizasyon algoritması ve Negatif Log Olabilirlik Kaybı (NLL Loss) kullanılarak eğitilmiştir. Eğitim sırasında model, her epoch'ta veri seti üzerinden geçerek ağırlıklarını güncellemiş ve kayıp (loss) değerini minimize etmeye çalışmıştır.
Test aşamasında model, hiç görmediği doğrulama verisi üzerinde denenmiş ve tahminleri gerçek etiketlerle karşılaştırılarak her sınıf için "Doğruluk (Accuracy)" oranları hesaplanmıştır. Sonuçlar `Results.csv` dosyasına kaydedilmiş ve çubuk grafiği (Bar Plot) ile görselleştirilmiştir.

---

## 5. SONUÇ VE DEĞERLENDİRME

Bu projede, derin öğrenme alanındaki modern mimarilerden biri olan Graf Sinir Ağları'nın (GNN), SAR (Synthetic Aperture Radar) görüntüleri üzerinde nasıl kullanılabileceğini öğrendim. SAR görüntülerini ızgara (grid) tabanlı yama (patch) yapılarına bölerek ve parçalar arasında komşuluk ilişkileri kurarak, pikseller arasındaki uzamsal bağımlılıkların GNN tarafından nasıl başarıyla modellenebileceğini uygulamalı olarak deneyimledim. 

Klasik CNN mimarileri görüntü işleme alanında standart kabul edilse de, GNN'lerin veriyi ilişkisel bir ağ (graf) olarak modelleme yeteneği, hedef sınıflandırma problemlerine çok daha esnek ve yenilikçi bir bakış açısı sunmaktadır. Proje sürecinde beni en çok zorlayan aşama, ham görüntü matrislerini graf formatına (düğüm özellikleri ve edge_index matrislerine) dönüştürmek oldu; ancak PyTorch Geometric kütüphanesinin sunduğu veri işleme araçları sayesinde bu sorunu çözdüm. 

Elde ettiğimiz doğruluk oranları, kurduğumuz 3 katmanlı basit GCN mimarisi göz önüne alındığında gayet tatmin ediciydi. Gelecekte bu başarı oranını daha da yukarı taşımak için modele GAT (Graph Attention Networks) gibi dikkat (attention) mekanizmaları eklenebilir veya veri artırma (data augmentation) yöntemleriyle model desteklenebilir. Bu proje, derin öğrenme ve graf teorisini birleştirme konusunda bana çok değerli pratik beceriler kazandırdı.

