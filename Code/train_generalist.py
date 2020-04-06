import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pickle
import torch.nn.functional as F
import json
import torch.optim as optim
import random
import os
import torchvision.models as models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
paramK = 10
def compute_confusion_matrix(classes,trueclass, pred):
    result = np.zeros((len(classes), paramK))
    for i in range(len(pred)):
        # print(trueclass[i])
        # print(pred[i])
        result[trueclass[i]][pred[i]] += 1
    return result

def get_new_specialties(classes,conf_matrix):
    C = len(classes)
    cselect = list(range(C))

    specialties = {}
    for sp in range(paramK):
        specialties[sp] = []
    lmapping = {}
    while len(cselect) > 0:
        randclass = random.choice(cselect)
        maxval = -100000000
        maxind = 0
        for j in range(paramK):
            if conf_matrix[randclass][j] > maxval and len(specialties[j]) < C/paramK:
                maxval = conf_matrix[randclass][j]
                maxind = j
        lmapping[randclass] = maxind
        specialties[maxind].append(randclass)
        cselect.remove(randclass)
    return lmapping, specialties
class CNNGeneralist(nn.Module):
    def __init__(self):
        super(CNNGeneralist, self).__init__()
        # self.features = nn.Sequential(
        #     # nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     # nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(kernel_size=3, stride=2),
        #     # nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     # nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(kernel_size=3, stride=2),
        #     # nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(kernel_size=3, stride=2),
        #
        #     # nn.LocalResponseNorm(2),
        #
        #     # nn.LocalResponseNorm(2),
        #
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(64 * 5, paramK)
        #     # nn.ReLU(inplace=True),
        #     # nn.Dropout(),
        #     # nn.Linear(100, 1),
        #     # nn.ReLU(inplace=True),
        #     # nn.Linear(1024, paramK),
        # )
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),  # w=28*28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13*13
            nn.LocalResponseNorm(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5),  # 9*9
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),# 4*4
            nn.LocalResponseNorm(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )
        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear(3136, paramK)
        # self.fc2 = nn.Linear(784, paramK)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        # out = F.relu(self.fc1(out))
        out = self.fc1(out)
        return out





def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.imshow(npimg)
    plt.show()



if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./cifar100', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR100(root='./cifar100', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=4)
    classes = pickle.load(open('./cifar100/cifar-100-python/meta', 'rb'))
    classes = classes['fine_label_names']
    trainsize = len(trainset)
    # print(trainsize)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = CNNGeneralist()
    # net = models.alexnet()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    #initializing specialties
    C = len(classes)
    K = paramK
    cselect = list(range(C))
    kselect = list(range(K))


    # Initializing Specialty labels l
    specialties = {}
    lmapping = {}
    count = 0
    while len(cselect) > 0:
        randclass = random.choice(cselect)
        randspec = random.choice(kselect)

        if randspec not in specialties:
            specialties[randspec] = []
        specialties[randspec].append(randclass)
        lmapping[randclass] = randspec
        if len(specialties[randspec]) == C / K:
            kselect.remove(randspec)
        cselect.remove(randclass)

    # print(specialties)
    # print(lmapping)
    PATH = './Model/cifar_net.pth'

    test = False
#
    if test:
        net.load_state_dict(torch.load(PATH))
        with open('./Model/specs.jsons', 'r') as f:
            specialties = json.load(f)
        for k, v in specialties.items():
            for l in v:
                lmapping[l] = int(k)
        print("Model loaded successfully!")
    else:
        print("Training Generalist...")
        for alternation in range(20):
            print(alternation, "alternation")
            running_loss = 0.0
            for epoch in range(10):  # loop over the dataset multiple times
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]

                    inputs, labels = data[0].to(device),torch.tensor([lmapping[itr] for itr in data[1].numpy()]).to(device)
                    # print(labels,data[1].numpy())
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    # print(i)
                    # if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d epochs] loss: %.3f' %
                    (epoch + 1, running_loss))
                        # running_loss = 0.0
            randsampler = list(range(0,trainsize))
            np.random.seed(42)
            np.random.shuffle(randsampler)
            randsampler = randsampler[: int(trainsize / 10)]
            subsetsampler = torch.utils.data.SubsetRandomSampler(randsampler)
            subsetloader = torch.utils.data.DataLoader(trainset, batch_size=5,
                                               sampler=subsetsampler)
            subsetiter = iter(subsetloader)
            correct = 0
            total = 0
            y_pred = []
            y_true = []
            with torch.no_grad():
                for data in subsetloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    labels = labels.cpu()
                    predicted = predicted.cpu()
                    # if len(y_pred) < 10:
                    #     print(labels.numpy())
                    #     print(predicted.numpy())
                    y_pred.extend([i for i in predicted.numpy()])
                    y_true.extend([i for i in labels.numpy()])
                    correct += (predicted == labels).sum().item()
            print("Getting Confusion Matrix")
            # print(len(y_true))
            # print(len(y_pred))
            # print(y_true)
            # print(y_pred)
            conf_matrix = compute_confusion_matrix(classes,y_true,y_pred)
            print("Confusion Matrix")
            print(conf_matrix)
            lmapping, specialties = get_new_specialties(classes, conf_matrix)
            correct = 0
            total = 0
            y_pred = []
            y_true = []
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    labels = labels.cpu()
                    predicted = predicted.cpu()
                    # if len(y_pred) < 10:
                    #     print(labels.numpy())
                    #     print(predicted.numpy())
                    y_pred.extend([i for i in predicted.numpy()])
                    y_true.extend([lmapping[i] for i in labels.numpy()])
                    convpred = torch.tensor([i for i in predicted.numpy()])
                    convtrue = torch.tensor([lmapping[i] for i in labels.numpy()])
                    correct += (convpred == convtrue).sum().item()
            print('Accuracy: %.2f %%' % (
                    100 * correct / total))
            # conf_matrix = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')
        print('Finished Training')
        print("Saving Model")
        PATH = './Model/cifar_net.pth'
        torch.save(net.state_dict(), PATH)
        with open('./Model/specs.jsons', 'w') as f:
            json.dump(specialties, f)

    print("Specialties: ",specialties)
    print("Label Mapping: ",lmapping)
    print("Testing Generalist")
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels = labels.cpu()
            predicted = predicted.cpu()
            # if len(y_pred) < 10:
            #     print(labels.numpy())
            #     print(predicted.numpy())
            y_pred.extend([i for i in predicted.numpy()])
            y_true.extend([lmapping[i] for i in labels.numpy()])
            convpred = torch.tensor([i for i in predicted.numpy()])
            convtrue = torch.tensor([lmapping[i] for i in labels.numpy()])
            correct += (convpred == convtrue).sum().item()

    print('Accuracy of the network on the 10000 test images: %.2f %%' % (
            100 * correct / total))

