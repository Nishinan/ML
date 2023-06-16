from model import OneStepNet, OneStepData
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

train_x = torch.load('fp_train.pt')
train_y = torch.load('index_train.pt')
train_product = torch.load('product_train.pt')
train_dataset = OneStepData(train_x, train_y)
val_x = torch.load('fp_val.pt')
val_y = torch.load('index_val.pt')
val_product = torch.load('product_val.pt')
val_dataset = OneStepData(val_x, val_y)
test_x = torch.load('fp_test.pt')
test_y = torch.load('index_test.pt')
test_product = torch.load('product_test.pt')
test_dataset = OneStepData(test_x, test_y)
print(len(train_x), len(train_y), len(train_product))
print(len(val_x), len(val_y), len(val_product))
print(len(test_x), len(test_y), len(test_product))
# 35830 35830 35940
# 3552 3552 4468
# 3577 3577 4487
# model = OneStepNet()
model = torch.load('onestep4.pt')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.01)
valid_loss_min = np.Inf  # track change in validation loss
tloss = []
vloss = []
acct = []
accv = []
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256)
for epoch in range(1, 11):
    train_loss = 0.0
    valid_loss = 0.0
    correct_train = 0
    cor_k = 0

    model.train()
    for data, target, target_index in train_loader:
        optimizer.zero_grad()
        # print(data[0])
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct_tensor = torch.eq(pred, target_index)
        correct = (correct_tensor == True).sum()
        correct_train += correct.item()
        # acct.append(correct_train)
        # top_kv, top_k = torch.topk(output, k=20, dim=1, largest=True, sorted=True)
        # # print(top_k[0:5])
        # # print(top_kv[0:5])
        # for i in range(len(target_index)):
        #     a = top_k[i] == target_index[i]
        #     if a.any():
        #         cor_k += 1
        # # print(correct, correct_tensor)
        # print(cor_k)
        # break
        # print(correct)

    class_correct = 0
    class_total = 0
    model.eval()
    correct_val = 0
    cor_kv = 0
    for data, target, target_index in val_loader:
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct_tensor = torch.eq(pred, target_index)
        correct = (correct_tensor == True).sum()
        correct_val += correct.item()
        # accv.append(correct_val)
        top_kv, top_k = torch.topk(output, k=20, dim=1, largest=True, sorted=True)
        # print(top_k[0:5])
        # print(top_kv[0:5])
        for i in range(len(target_index)):
            a = top_k[i] == target_index[i]
            if a.any():
                cor_kv += 1
        # correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.numpy())
        # for i in range(len(target.data)):
        #     label = target.data[i]
        #     class_correct += correct[i].item()
        #     class_total += 1

    # 计算平均损失
    train_loss = train_loss / len(train_dataset)
    valid_loss = valid_loss / len(val_dataset)
    tloss.append(train_loss)
    vloss.append(valid_loss)
    acct.append(cor_k)
    accv.append(cor_kv)

    # 显示训练集与验证集的损失函数
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    print(correct_train, correct_val, cor_k, cor_kv)
    class_correct = 0
    class_total = 0
print(tloss, vloss, acct, accv)

model.eval()
correct_test = 0
cor_kt = 0
for data, target, target_index in test_loader:
    output = model(data)
    loss = criterion(output, target)
    _, pred = torch.max(output, 1)
    correct_tensor = torch.eq(pred, target_index)
    correct = (correct_tensor == True).sum()
    correct_test += correct.item()
    top_kv, top_k = torch.topk(output, k=200, dim=1, largest=True, sorted=True)
    # print(top_k[0:5])
    # print(top_kv[0:5])
    for i in range(len(target_index)):
        a = top_k[i] == target_index[i]
        if a.any():
            cor_kt += 1

print(correct_test, cor_kt)
torch.save(model, 'onestep5.pt')