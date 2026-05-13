import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNNClassifier, self).__init__()
        
        # Graph Convolutional Katmanları
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        
        # Tam Bağlı (Fully Connected) Katman
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. GCN Katmanı
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # 2. GCN Katmanı
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 3. GCN Katmanı
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global Pooling (Her grafın düğümlerinin ortalamasını alarak tüm grafı tek vektör yapar)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # Sınıflandırma
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
