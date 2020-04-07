import pickle
import json

classes=pickle.load(open('../cifar100/cifar-100-python/meta', 'rb'))
classes=classes['fine_label_names']
specs = 0
with open('../Model/specs.jsons','r') as f:
    specs = json.load(f)

for k, v in specs.items():
    print("Speciality ",k,": ",[classes[i] for i in v])