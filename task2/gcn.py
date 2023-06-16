import numpy as np
import pickle
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import combinations
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt


# Load data and labels
with open('train.pkl', 'rb') as f:
    train_data = pickle.load(f)
    
with open('test.pkl', 'rb') as f:
    test_data = pickle.load(f)

train_X = np.unpackbits(train_data['packed_fp'], axis=1)
train_y = train_data['values']

test_X = np.unpackbits(test_data['packed_fp'], axis=1)
test_y = test_data['values']

train_samples = train_X.shape[0]
test_samples = test_X.shape[0]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_similarity(fp1, fp2):
    dot_product = np.dot(fp1, fp2)
    norm_fp1 = np.linalg.norm(fp1)
    norm_fp2 = np.linalg.norm(fp2)
    similarity = dot_product / (norm_fp1 * norm_fp2)
    return similarity

# Create graphs for training set
train_graphs = []
for group in np.array_split(np.arange(train_samples), train_samples // 300):
    group_X = train_X[group]
    group_y = train_y[group]
    num_samples = group_X.shape[0]
    adj_matrix = np.zeros((num_samples, num_samples))
    num_edges = 0
    for i, j in combinations(range(num_samples), 2):
        similarity = compute_similarity(group_X[i], group_X[j])
        if similarity > 0.6:
            adj_matrix[i, j] = similarity
            adj_matrix[j, i] = similarity
            num_edges += 1

    if num_edges > 160:
        coo = coo_matrix(adj_matrix)
        edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long).to(device)
        edge_attr = torch.tensor(coo.data, dtype=torch.float).to(device)
        train_graphs.append(Data(x=torch.tensor(group_X, dtype=torch.float).to(device),
                                y=torch.tensor(group_y, dtype=torch.float).to(device),
                                edge_index=edge_index,
                                edge_attr=edge_attr))

# Create graphs for test set
test_graphs = []
for group in np.array_split(np.arange(test_samples), test_samples // 300):
    group_X = test_X[group]
    group_y = test_y[group]
    num_samples = group_X.shape[0]
    adj_matrix = np.zeros((num_samples, num_samples))
    num_edges = 0
    for i, j in combinations(range(num_samples), 2):
        similarity = compute_similarity(group_X[i], group_X[j])
        if similarity > 0.6:
            adj_matrix[i, j] = similarity
            adj_matrix[j, i] = similarity
            num_edges += 1

    if num_edges > 160:
        coo = coo_matrix(adj_matrix)
        edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long).to(device)
        edge_attr = torch.tensor(coo.data, dtype=torch.float).to(device)
        test_graphs.append(Data(x=torch.tensor(group_X, dtype=torch.float).to(device),
                                y=torch.tensor(group_y, dtype=torch.float).to(device),
                                edge_index=edge_index,
                                edge_attr=edge_attr))

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x

# Define the custom dataset
class MolDataset(Dataset):
    def __init__(self, graphs):
        super(MolDataset, self).__init__()
        self.graphs = graphs

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

# Create data loaders
train_dataset = MolDataset(train_graphs)
test_dataset = MolDataset(test_graphs)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, optimizer, and loss function
model = GCN(in_channels=train_X.shape[1], hidden_channels=64, out_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
def train(model, optimizer, criterion, train_loader):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()

# Evaluation
def evaluate(model, criterion, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out, data.y.view(-1, 1))
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(data_loader.dataset)

# Train the model
num_epochs = 50
train_losses = []
test_losses = []
best_test_loss = float('inf')
best_model_params = None
for epoch in range(num_epochs):
    train(model, optimizer, criterion, train_loader)
    train_loss = evaluate(model, criterion, train_loader)
    test_loss = evaluate(model, criterion, test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_params = copy.deepcopy(model.state_dict()) 
 
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss2.png')
plt.show() 

torch.save(best_model_params, 'best_model2.pth')
model_best = GCN(in_channels=train_X.shape[1], hidden_channels=64, out_channels=1).to(device)
model_best.load_state_dict(torch.load('best_model2.pth'))
model_best.eval()

predictions = []
targets = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model_best(data.x, data.edge_index, data.edge_attr)
        predictions.append(out.detach().cpu().numpy())
        targets.append(data.y.view(-1, 1).cpu().numpy())
predictions = np.concatenate(predictions)
targets = np.concatenate(targets)


mse = mean_squared_error(targets, predictions)
mae = mean_absolute_error(targets, predictions)
r2 = r2_score(targets, predictions)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2: {r2:.4f}")

