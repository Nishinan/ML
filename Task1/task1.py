import numpy as np
import torch

from rdchiral.template_extractor import extract_from_reaction
from rdkit import Chem
from rdkit.Chem import AllChem
from rdchiral.main import rdchiralRunText

data = np.loadtxt("./schneider50k/raw_test.csv", delimiter=",", dtype=str)
reactions = data[1:, 2]
print(len(reactions))
w_react = 0
i = 0
i1 = 0
# tem = torch.load('template.pt')
# a = "[#7:6]-[C@@H;D3;+0:5]1-[C:4]-[C:3]-[C@@H;D3;+0:1](-[c:2])-[C:8]-[C:7]-1>>O-[C;H0;D4;+0:1]1(-[c:2])-[C:3]-[C:4]-[CH;D3;+0:5](-[#7:6])-[C:7]-[C:8]-1"
# indexa = tem.index(a) if a in tem else -1
# print(indexa)
# tem_set = list(set(tem))
# torch.save(tem_set, 'tem_set.pt')
# print(tem[0])
tem_set = torch.load('tem_set.pt')
print(tem_set[0])
prod, idx = [], []
fps = []
# print(len(tem), len(tem_set))
for reaction in reactions:
    i += 1
    # reaction=reactions[143]
    try:
        reactants, products = reaction.split('>>')
        products_ = products.split('.')
        for product in products_:
            inputRec = {'_id': None, 'reactants': reactants, 'products': product}
            ans = extract_from_reaction(inputRec)
            if 'err_msg' in ans.keys():
                print(ans['err_msg'], i)
            if 'reaction_smarts' in ans.keys():
                # print('o')
                i1 += 1
                prod.append(product)
                # print(i1, product)
                # print(i1, len(prod))
                index = tem_set.index(ans['reaction_smarts']) if ans['reaction_smarts'] in tem_set else -1
                # if index != -1:
                #     print(i, 'wrong')
                #     print('r', type(reactants), reactants)
                #     print('p', type(product), product)
                #     print('t', type(ans['reaction_smarts']), ans['reaction_smarts'])
                print(i1, index)
                idx.append(index)
                # print(i1, idx)
                mol = Chem.MolFromSmarts(product)
                mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                # print(type(fp))
                onbits = list(fp.GetOnBits())
                arr = np.zeros(fp.GetNumBits())
                arr[onbits] = 1
                # print(len(prod))
                # print(len(idx))
                fps.append(arr)
                # print(i1, len(fps))

                print(tem_set[index], index)
                print(ans['reaction_smarts'])
                out = rdchiralRunText(ans['reaction_smarts'], product)
                # print(i)
                print('out', type(out), out)
                print('r', type(reactants), reactants)
                print('p', type(product), product)
                print('t', type(ans['reaction_smarts']), ans['reaction_smarts'])
                print()
    except Exception as e:
        w_react += 1
        # print(e)
    # break
print(w_react)
# torch.save(prod, 'product_test1.pt')
# torch.save(fps, 'fp_test1.pt')
# torch.save(idx, 'index_test1.pt')
# tem_set = set(tem)
# print(len(tem), len(tem_set))
# 4035
# test 5007 1435 3572
# 35939 10562
# torch.save(tem, 'template.pt')
