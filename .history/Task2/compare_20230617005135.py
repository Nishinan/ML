import numpy as np
import pickle
import random
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Dataset, DataLoader
from itertools import combinations
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt


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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims)-1)])
        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        x = self.fc_out(x)
        return x

class MolDataset(Dataset):
    def __init__(self, graphs):
        super(MolDataset, self).__init__()
        self.graphs = graphs

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


test_dataset = MolDataset(test_graphs)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model_best = GCN(in_channels=train_X.shape[1], hidden_channels=64, out_channels=1).to(device)
model_best.load_state_dict(torch.load('best_model2.pth'))
model_best.eval()

model_best_single = MLP(input_dim=train_X.shape[1], hidden_dims=[256, 256, 256], output_dim=1).to(device)
model_best_single.load_state_dict(torch.load('best_model1.pth'))
model_best_single.eval()

predictions = []
targets = []
single_predictions = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model_best(data.x, data.edge_index, data.edge_attr)
        predictions.append(out.detach().cpu().numpy())
        targets.append(data.y.view(-1, 1).cpu().numpy())

        single_out = model_best_single(data.x)
        single_predictions.extend(single_out.squeeze().cpu().numpy())

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)

single_predictions = np.array(single_predictions)

selected_groups = random.sample(range(len(predictions)), 1000)
multiple_sums = []
single_sums = []
true_sums = []
for group_index in selected_groups:
    multiple_sum = np.sum(predictions[group_index])
    multiple_sums.append(multiple_sum)

    single_sum = np.sum(single_predictions[group_index])
    single_sums.append(single_sum)
    
    true_sum = np.sum(targets[group_index])
    true_sums.append(true_sum)

multiple_sums = np.array(multiple_sums)
single_sums = np.array(single_sums)
true_sums = np.array(true_sums)
multiple_errors = np.abs(multiple_sums - true_sums)
single_errors = np.abs(single_sums - true_sums)

plt.hist(multiple_errors, bins=10, alpha=0.5, label='GCN Model')
plt.hist(single_errors, bins=10, alpha=0.5, label='MLP Model')
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("compare.png")
plt.show()


multiple_mean_error = np.mean(multiple_errors)
single_mean_error = np.mean(single_errors)
print("GCN Model - Mean Absolute Error:", multiple_mean_error)
print("MLP Model - Mean Absolute Error:", single_mean_error)