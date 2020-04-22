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

paramK = 10
def compute_confusion_matrix(classes,trueclass, pred):
    result = np.zeros((len(classes), paramK))
    for i in range(len(pred)):
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

class CNNGeneralistResNet(nn.Module):
    def __init__(self):
        super(CNNGeneralistResNet, self).__init__()
        resnet = models.resnet50()
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # self.fc1 = nn.Linear(65536,4096)
        self.drop_out = nn.Dropout()
        self.fc2 = nn.Linear(512, paramK)

    def forward(self, x):
        out = self.resnet(x)
        out = out.reshape(out.size(0), -1)
        # out = torch.transpose(x,0,1)
        out = self.drop_out(out)
        # out = self.fc1(out)
        out = self.fc2(out)
        return out

class CNNGeneralist(nn.Module):
    def __init__(self):
        super(CNNGeneralist, self).__init__()
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

        self.fc1 = nn.Linear(7744, paramK)
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
        # transforms.Resize(160),
        transforms.CenterCrop(28),
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
    net = None
    flag = "res"
    if flag == "c100":
        net = CNNGeneralist()

    elif flag == "alex":
        net = models.alexnet()
        classifier = list(net.classifier.children())
        net.classifier = nn.Sequential(*classifier[:-1])
        net.classifier.add_module(
            '6', nn.Linear(classifier[-1].in_features, paramK))
    elif flag == "res":
        net = CNNGeneralistResNet()

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


    test = False
    PATH = './Model/cifar_net.pth'
    if flag == "alex":
        PATH = './Model/alex_gen.pth'
    elif flag == 'res':
        PATH = './Model/resnet_gen.pth'
    PATHspec = './Model/specs.jsons'
    if flag == "alex":
        PATHspec = './Model/alexspecs.jsons'
    elif flag == 'res':
        PATHspec = './Model/resnetspecs.jsons'

    if test:
        net.load_state_dict(torch.load(PATH))
        with open(PATHspec, 'r') as f:
            specialties = json.load(f)
        for k, v in specialties.items():
            for l in v:
                lmapping[l] = int(k)

        print("Model loaded successfully!")
    else:
        print("Training Generalist...")
        for alternation in range(15):
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

        torch.save(net.state_dict(), PATH)

        with open(PATHspec, 'w') as f:
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

