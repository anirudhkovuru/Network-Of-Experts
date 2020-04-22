import pickle
import json

classes=pickle.load(open('./cifar100/cifar-100-python/meta', 'rb'))
classes=classes['fine_label_names']
specs = 0
flag = "alex"
PATH = './results1/specs.jsons'
# if flag == "alex":
#     PATH = './Model/alexspecs.jsons'
# elif flag == "res":
#     PATH = './Model/resnetspecs.jsons'
with open(PATH,'r') as f:
    specs = json.load(f)

for k, v in specs.items():
    print("Speciality ",k,": ",[classes[i] for i in v])