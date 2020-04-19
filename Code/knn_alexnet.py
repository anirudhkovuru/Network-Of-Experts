import numpy as np
import torchvision.models as models
import torch.utils.data.dataloader
import torchvision
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    alexnet_model = models.alexnet(pretrained=True)
    alexnet_model.to(device)

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the test dataset
    test_set = torchvision.datasets.CIFAR100(root='./cifar100', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    feature_vectors = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            alexnet_output = alexnet_model.features[0](images)
            for i in range(1, 13):
                alexnet_output = alexnet_model.features[i](alexnet_output)
            for i, label in enumerate(labels):
                feature_vectors.append(alexnet_output[i].cpu().numpy().flatten())

    feature_vectors = np.array(feature_vectors)
    neighbours = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(feature_vectors)
    _, indices = neighbours.kneighbors(feature_vectors)

    np.save('../Model/nn_base.npy', indices)
