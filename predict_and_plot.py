import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from dataset import SARGraphDataset
from model import GNNClassifier

def predict_and_plot():
    data_dir = os.path.join(os.path.dirname(__file__), 'MSTAR_Dummy') # Gerçek veriseti varsa burayı değiştirin
    
    print("Test veriseti yükleniyor...")
    test_dataset = SARGraphDataset(root=data_dir, split='test')
    
    if len(test_dataset) == 0:
        print("Test veriseti bulunamadı!")
        return

    # Sınıf isimlerini al
    classes_file = os.path.join(data_dir, 'processed', 'classes_test.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
    else:
        classes = sorted(os.listdir(os.path.join(data_dir, 'raw', 'test')))
        
    num_classes = len(classes)
    num_node_features = 64
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNClassifier(num_node_features=num_node_features, num_classes=num_classes).to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), 'gnn_model.pth')
    if not os.path.exists(model_path):
        print(f"Model dosyası bulunamadı: {model_path}. Lütfen önce train.py çalıştırın.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    results = []
    
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    print("Tahminler yapılıyor...")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1).item()
            true_label = data.y.item()
            
            # Doğruluk hesaplama için kaydet
            if pred == true_label:
                class_correct[true_label] += 1
            class_total[true_label] += 1
            
            results.append({
                'Image_Index': i,
                'True_Class': classes[true_label],
                'Predicted_Class': classes[pred],
                'Is_Correct': pred == true_label
            })
            
    # CSV'ye kaydet
    df = pd.DataFrame(results)
    csv_path = os.path.join(os.path.dirname(__file__), 'Results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Sonuçlar {csv_path} dosyasına kaydedildi.")
    
    # Grafikleri çizdir (Doğruluk Oranı Çubuk Grafiği)
    accuracies = []
    plot_classes = []
    
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            accuracies.append(acc)
            plot_classes.append(classes[i])
            print(f'Sınıf {classes[i]} Doğruluğu: % {acc:.2f}')
    
    plt.figure(figsize=(10, 6))
    plt.bar(plot_classes, accuracies, color='skyblue')
    plt.xlabel('Hedef Sınıfları')
    plt.ylabel('Doğruluk Oranı (%)')
    plt.title('SAR Görüntüleri GNN Sınıflandırma Doğruluğu')
    plt.ylim(0, 110)
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
        
    plot_path = os.path.join(os.path.dirname(__file__), 'accuracy_plot.png')
    plt.savefig(plot_path)
    print(f"Doğruluk grafiği {plot_path} olarak kaydedildi.")
    plt.show()

if __name__ == '__main__':
    predict_and_plot()
