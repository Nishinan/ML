from model import OneStepNet, OneStepData
import numpy as np
import torch 
from rdkit import Chem
from rdkit.Chem import AllChem
from rdchiral.main import rdchiralRunText

test_x = torch.load('fp_test1.pt')
test_y = torch.load('index_test1.pt')
test_product = torch.load('product_test1.pt')
tem_set = torch.load('tem_set.pt')
print(len(test_x), len(test_y), len(test_product))
# print(test_y[3570:])
model = torch.load('onestep4.pt')
num = 0
for i in range(len(test_x)):
    x = torch.tensor(test_x[i])
    output = model(x)
    output = torch.unsqueeze(output, dim=1)
    f = 0
    # print(output.shape)
    _, pred = torch.max(output, 0)
    correct_tensor = torch.eq(pred, test_y[i])
    correct = (correct_tensor == True).sum()
    top_kv, top_k = torch.topk(output, k=200, dim=0, largest=True, sorted=True)
    # if correct == 1:
    #     print(i)
    # print(pred)
    reactants_real = []
    print(test_product[i])
    try:
        reactants_real = rdchiralRunText(tem_set[test_y[i]], test_product[i])
        if len(reactants_real) != 0:
            print(i, 'real', test_y[i], num, tem_set[test_y[i]], reactants_real)
    except Exception as e:
        print(e)
    for pred in top_k:
        try:
            reactants = rdchiralRunText(tem_set[pred], test_product[i])
            if len(reactants) != 0:
                print(i, 'pred', pred, num, tem_set[pred], reactants)
                if reactants_real == reactants:
                    print(i, pred)
                    f = 1
        except Exception as e:
            print(e)
    num += f

    print()
    # print(top_kv, top_k)
    # print(test_y[i])
    # success 1175
