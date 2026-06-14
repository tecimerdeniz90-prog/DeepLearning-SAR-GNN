import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from dataset import SARGraphDataset
from model import GNNClassifier

def train():
    data_dir = os.path.join(os.path.dirname(__file__), 'MSTAR_Dummy') # Gerçek veriseti varsa burayı değiştirin
    
    print("Eğitim veriseti yükleniyor...")
    train_dataset = SARGraphDataset(root=data_dir, split='train', apply_noise=True)
    
    if len(train_dataset) == 0:
        print("Eğitim veriseti bulunamadı! Lütfen önce generate_dummy_data.py dosyasını çalıştırın.")
        return

    num_classes = len(os.listdir(os.path.join(data_dir, 'raw', 'train')))
    
    # 8x8 patch kullanıyoruz, her patch 8x8 pixel (64 boyutlu vektör)
    num_node_features = 64 
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNClassifier(num_node_features=num_node_features, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    model.train()
    epochs = 20
    
    loss_history = []
    acc_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            
        epoch_loss = total_loss / len(train_dataset)
        acc = correct / len(train_dataset)
        
        loss_history.append(epoch_loss)
        acc_history.append(acc)
        
        print(f'Epoch {epoch+1:03d}, Loss: {epoch_loss:.4f}, Train Acc: {acc:.4f}')

    # Eğitim Grafiğini Çizdir ve Kaydet
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Train Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(acc_history, label='Train Accuracy', color='orange')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(__file__), 'training_history.png')
    plt.savefig(plot_path)
    print(f"Eğitim grafiği kaydedildi: {plot_path}")

    # Modeli kaydet
    model_save_path = os.path.join(os.path.dirname(__file__), 'gnn_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model {model_save_path} konumuna kaydedildi.")

if __name__ == '__main__':
    train()
