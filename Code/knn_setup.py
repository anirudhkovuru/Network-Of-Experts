import json

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch.utils.data.dataloader
import torchvision
from torchvision.transforms import transforms

from Code.train_experts import CNNExperts
from Code.train_experts import CNNGeneralist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    C = 100
    K = 10

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the test dataset
    test_set = torchvision.datasets.CIFAR100(root='./cifar100', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    # Get the mapping
    lmapping = {}
    with open('../Model/specs.jsons', 'r') as f:
        specialties = json.load(f)
    for k, v in specialties.items():
        for label in v:
            lmapping[label] = int(k)

    saved_models = torch.load("../Model/nofe.pth")
    expert_models = {}
    trunk_counter = 0
    layer_counter = 0
    class_counter = 0
    for (k, v) in saved_models['experts'].items():
        if trunk_counter < 6:
            trunk_counter += 1
            continue

        if layer_counter == 0:
            expert_models[class_counter] = {}
            expert_models[class_counter]["layer1.0.weight"] = v
        if layer_counter == 1:
            expert_models[class_counter]["layer1.0.bias"] = v
        if layer_counter == 2:
            expert_models[class_counter]["fc1.weight"] = v
        if layer_counter == 3:
            expert_models[class_counter]["fc1.bias"] = v
        layer_counter += 1
        if layer_counter == 4:
            layer_counter = 0
            class_counter += 1

    # Prepare generalist
    generalist_net = CNNGeneralist()
    generalist_net.load_state_dict(saved_models['gen'])
    generalist_net.to(device)

    # Prepare the experts
    experts = []
    for i in range(K):
        experts.append(CNNExperts(C))
        save = {}
        experts[i].load_state_dict(expert_models[i])
        experts[i].to(device)

    feature_vectors = []
    feature_labels = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            gen_output = generalist_net(images)
            expert_outputs = []
            for exp in experts:
                expert_outputs.append(exp.layer1(gen_output))

            for i, label in enumerate(labels):
                k = lmapping[int(label.cpu().numpy())]
                feature_vectors.append(expert_outputs[k][i].cpu().numpy().flatten())
                feature_labels.append(int(label.cpu().numpy()))

    feature_vectors = np.array(feature_vectors)
    neighbours = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(feature_vectors)
    _, indices = neighbours.kneighbors(feature_vectors)

    np.save('../Model/nn_experiment.npy', indices)
