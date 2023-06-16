import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import copy
from matplotlib import pyplot as plt

with open('train.pkl', 'rb') as f:
    train_data = pickle.load(f)
    
with open('test.pkl', 'rb') as f:
    test_data = pickle.load(f)

train_X = np.unpackbits(train_data['packed_fp'], axis=1)
train_y = train_data['values']

test_X = np.unpackbits(test_data['packed_fp'], axis=1)
test_y = test_data['values']

test_y = test_y.numpy()


class MoleculeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = MoleculeDataset(train_X, train_y)
test_dataset = MoleculeDataset(test_X, test_y)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_dims = [256, 256, 256]
model = MLP(input_dim=train_X.shape[1], hidden_dims=hidden_dims, output_dim=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
train_losses = []
test_losses = []
best_test_loss = float('inf')
best_model_params = None

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for features, labels in train_loader:
        features = features.float().to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * features.size(0)

    train_loss /= len(train_dataset)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.float().to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * features.size(0)

    test_loss /= len(test_dataset)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_params = copy.deepcopy(model.state_dict()) 

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss1.png")
plt.show()

torch.save(best_model_params, 'best_model1.pth')

model_best = MLP(input_dim=train_X.shape[1], hidden_dims=hidden_dims, output_dim=1).to(device)
model_best.load_state_dict(torch.load('best_model1.pth'))
model_best.eval()

predictions = []
with torch.no_grad():
    for features, labels in test_loader:
        features = features.float().to(device)
        outputs = model_best(features)
        predictions.extend(outputs.squeeze().cpu().numpy())

predictions = np.array(predictions)

mse = mean_squared_error(test_y, predictions)
mae = mean_absolute_error(test_y, predictions)
r2 = r2_score(test_y, predictions)
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2: {r2:.4f}")
