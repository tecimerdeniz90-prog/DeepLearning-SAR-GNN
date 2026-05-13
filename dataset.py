import os
import glob
import torch
import numpy as np
from PIL import Image
from torch_geometric.data import Data, InMemoryDataset
from torchvision import transforms

def image_to_graph(img_path, grid_size=8):
    """
    Bir görüntüyü Grid (Izgara) tabanlı bir grafa dönüştürür.
    Görüntü grid_size x grid_size parçaya (patch) bölünür.
    Her parça bir düğüm (node) olur. Düğüm özellikleri o parçanın piksel değerleridir.
    Komşu parçalar birbirine kenarlar (edges) ile bağlanır.
    """
    img = Image.open(img_path).convert('L') # SAR genelde gri seviyedir (grayscale)
    img = img.resize((64, 64)) # Görüntüyü standart bir boyuta getir
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    patch_h = img_array.shape[0] // grid_size
    patch_w = img_array.shape[1] // grid_size
    
    nodes = []
    edges_src = []
    edges_dst = []
    
    # Düğümleri oluştur
    for i in range(grid_size):
        for j in range(grid_size):
            patch = img_array[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            nodes.append(patch.flatten())
            
            node_idx = i * grid_size + j
            
            # Kenarları oluştur (4'lü komşuluk)
            if i > 0: # Yukarı komşu
                edges_src.append(node_idx)
                edges_dst.append((i-1) * grid_size + j)
            if i < grid_size - 1: # Aşağı komşu
                edges_src.append(node_idx)
                edges_dst.append((i+1) * grid_size + j)
            if j > 0: # Sol komşu
                edges_src.append(node_idx)
                edges_dst.append(i * grid_size + (j-1))
            if j < grid_size - 1: # Sağ komşu
                edges_src.append(node_idx)
                edges_dst.append(i * grid_size + (j+1))
                
    x = torch.tensor(np.array(nodes), dtype=torch.float)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)


class SARGraphDataset(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.split] # Beklenen raw klasör adı ('train' veya 'test')

    @property
    def processed_file_names(self):
        return [f'data_{self.split}.pt']

    def process(self):
        data_list = []
        split_dir = os.path.join(self.raw_dir, self.split)
        
        if not os.path.exists(split_dir):
            print(f"Uyarı: {split_dir} bulunamadı. Lütfen verisetini bu dizine kopyalayın.")
            return

        classes = sorted(os.listdir(split_dir))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        # Sınıf isimlerini kaydet (daha sonra tahmin grafiğinde kullanmak için)
        with open(os.path.join(self.processed_dir, f'classes_{self.split}.txt'), 'w') as f:
            for cls_name in classes:
                f.write(f"{cls_name}\n")

        for cls_name in classes:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
                
            img_paths = glob.glob(os.path.join(cls_dir, '*.png')) + \
                        glob.glob(os.path.join(cls_dir, '*.jpg')) + \
                        glob.glob(os.path.join(cls_dir, '*.jpeg'))
            
            for img_path in img_paths:
                data = image_to_graph(img_path, grid_size=8)
                data.y = torch.tensor([class_to_idx[cls_name]], dtype=torch.long)
                data_list.append(data)
                
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
