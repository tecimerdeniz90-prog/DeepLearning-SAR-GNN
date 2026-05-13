import os
import cv2
import numpy as np
from PIL import Image

def generate_dummy_mstar_data(base_path, num_classes=3, samples_per_class_train=20, samples_per_class_test=5, img_size=(128, 128)):
    """
    MSTAR veriseti yapısına benzeyen, test edebilmek amaçlı sahte bir veriseti oluşturur.
    """
    for split in ['train', 'test']:
        samples = samples_per_class_train if split == 'train' else samples_per_class_test
        for c in range(num_classes):
            class_name = f'Target_{c+1}'
            dir_path = os.path.join(base_path, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
            
            for i in range(samples):
                # Siyah arkaplan üzerine rastgele şekiller çizerek "radar hedefi" gibi sahte veriler yapalım
                img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
                
                # Hedefin sınıfına göre farklı şekiller
                center_x = np.random.randint(40, 88)
                center_y = np.random.randint(40, 88)
                
                if c == 0:
                    cv2.circle(img, (center_x, center_y), np.random.randint(10, 20), (255, 255, 255), -1)
                elif c == 1:
                    cv2.rectangle(img, (center_x-15, center_y-10), (center_x+15, center_y+10), (255, 255, 255), -1)
                else:
                    pts = np.array([[center_x, center_y-20], [center_x-15, center_y+10], [center_x+15, center_y+10]], np.int32)
                    cv2.fillPoly(img, [pts], (255, 255, 255))
                
                # Gürültü (Speckle noise) ekle, SAR görüntülerindeki gibi
                noise = np.random.randint(0, 50, (img_size[0], img_size[1], 3), dtype=np.uint8)
                img = cv2.add(img, noise)
                
                img_path = os.path.join(dir_path, f'img_{i}.png')
                Image.fromarray(img).save(img_path)
    
    print(f"Dummy veriseti '{base_path}' dizininde oluşturuldu.")

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), 'MSTAR_Dummy')
    generate_dummy_mstar_data(data_dir)
