import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import json
import torch.optim as optim
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
paramK = 10

'''class cnngeneralist(nn.Module):
        def __init__(self):
            super(cnngeneralist, self).__init__()
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

            self.fc1 = nn.Linear(12800, 784)
            self.fc2 = nn.Linear(784, paramK)

        def forward(self, x):
            return out


class cnnexpert
'''
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

        self.fc1 = nn.Linear(3136, paramK)
        # self.fc2 = nn.Linear(784, paramK)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

class CNNExperts(nn.Module):
    def __init__(self, num_classes):
        super(CNNExperts, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5),  # w=28*28
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),  # 13*13
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(64, int(num_classes/paramK))

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        # print(out.shape)
        return out

def softmax(scores):
  return np.exp(scores)/sum(np.exp(scores), axis=0)

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
    classes = list(range(100))
    trainsize = len(trainset)
    # print(trainsize)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    classes = pickle.load(open('./cifar100/cifar-100-python/meta', 'rb'))
    classes = classes['fine_label_names']

    C = len(classes)

    lmapping = {}
    specialties = {}

    net = CNNGeneralist()
    PATH = './cifar_net.pth'

    net.load_state_dict(torch.load(PATH))
    with open('./Model/specs.jsons', 'r') as f:
        specialties = json.load(f)
    for k, v in specialties.items():
        for l in v:
            lmapping[l] = int(k)
    print("Generalist Model loaded successfully!")
    # net = models.alexnet()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    gen_optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    experts = []
    optims = []
    for exp in range(paramK):
        experts.append(CNNExperts(C))
        experts[exp].to(device)
        optims.append(optim.SGD(experts[exp].parameters(), lr=0.001, momentum=0.9))
    test = False

    if test:
        PATH2 = './Model/nofe.pth'
        checkpoint = torch.load(PATH)
        net.load_state_dict(checkpoint['gen'])
        for i, exp in enumerate(experts):
            exp.load_state_dict(checkpoint[i])
    else:
        print("Training Experts...")
        running_loss = 0.0
        for epoch in range(30):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]

                inputs, labels = data[0].to(device), torch.tensor([lmapping[itr] for itr in data[1].numpy()]).to(device)
                # print(labels,data[1].numpy())
                # zero the parameter gradients
                gen_optimizer.zero_grad()
                for exp_optims in optims:
                    exp_optims.zero_grad()


                # forward + backward + optimize
                outputs = net(inputs)
                expout = []
                for expert in experts:
                    expout.append(expert(outputs))
                expout = torch.cat(expout, dim=1)
                # print(outputs)
                # for out in expout:
                #     loss += criterion(data[1],out)
                # loss.backward()
                # gen_optimizer.step()
                exploss = criterion(expout, data[1].to(device))
                exploss.backward()
                for i, optim in enumerate(optims):
                    # print(data[1].shape)
                    # print(expout[i].shape)
                    optim.step()

                # print statistics
                running_loss += exploss.item()
                # print(i)
                # if i % 2000 == 1999:  # print every 2000 mini-batches

            print('[%d epochs] loss: %.3f' %
                  (epoch + 1, running_loss))
            correct = 0
            total = 0
            y_pred = []
            y_true = []
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    expoutputs = []
                    for exp in experts:
                        expoutputs.append(exp(outputs))
                    expoutputs = torch.cat(expoutputs, dim=1)
                    # expoutputs = np.array(expoutputs)

                    _, predicted = torch.max(expoutputs, 1)
                    total += labels.size(0)

                    correct += (predicted == labels).sum().item()

            print('Accuracy: %.2f %%' % (
                    100 * correct / total))
        # running_loss = 0.0
        print("Training of NofE Done!")
        print("Saving Model")
        PATH = './Model/nofe.pth'
        saved = {}
        saved['gen'] = net.state_dict()
        for i,exp in enumerate(experts):
            saved[i] = exp.state_dict()
        torch.save(saved, PATH)
        print("Model has been saved!")
    print("Testing NofE...")
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            expoutputs = []
            for exp in experts:
                expoutputs.append(exp(outputs))
            expoutputs = torch.cat(expoutputs, dim=1)
            # expoutputs = np.array(expoutputs)

            _, predicted = torch.max(expoutputs, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.2f %%' % (
            100 * correct / total))
