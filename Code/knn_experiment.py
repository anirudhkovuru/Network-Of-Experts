import json

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image

CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

CIFAR100_LABELS_DICT = {}
for i, label in enumerate(CIFAR100_LABELS_LIST):
    CIFAR100_LABELS_DICT[label] = i


if __name__ == '__main__':
    nofe_neighbors = np.load('../Model/nn_experiment.npy')
    base_neighbors = np.load('../Model/nn_base.npy')
    test_set = torchvision.datasets.CIFAR100(root='./cifar100', train=False, download=True)

    lmapping = {}
    with open('../Model/specs.jsons', 'r') as f:
        specialties = json.load(f)
    for k, v in specialties.items():
        for label in v:
            lmapping[label] = int(k)

    total = 5
    start = 5
    places = [k for k in range(1, total*3, 3)]
    for i in range(start, start+total):
        plt.subplot(total, 3, places[i-start])
        # plt.subplot(1, 3, 1)
        plt.title(CIFAR100_LABELS_LIST[test_set[i][1]] + "(" + str(lmapping[test_set[i][1]]) + ")")
        plt.axis('off')
        plt.imshow(test_set[i][0])

        images = [test_set[nofe_neighbors[i][1]][0], test_set[nofe_neighbors[i][2]][0],
                  test_set[nofe_neighbors[i][3]][0]]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        plt.subplot(total, 3, places[i-start]+1)
        # plt.subplot(1, 3, 2)
        l1 = test_set[nofe_neighbors[i][1]][1]
        l2 = test_set[nofe_neighbors[i][2]][1]
        l3 = test_set[nofe_neighbors[i][3]][1]
        plt.title(CIFAR100_LABELS_LIST[l1] + "(" + str(lmapping[l1]) + ") " + CIFAR100_LABELS_LIST[l2] +
                  "(" + str(lmapping[l2]) + ") " + CIFAR100_LABELS_LIST[l3] + "(" + str(lmapping[l3]) + ")")
        plt.axis('off')
        plt.imshow(new_im)

        images = [test_set[base_neighbors[i][1]][0], test_set[base_neighbors[i][2]][0],
                  test_set[base_neighbors[i][3]][0]]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        plt.subplot(total, 3, places[i-start]+2)
        # plt.subplot(1, 3, 3)
        l1 = test_set[base_neighbors[i][1]][1]
        l2 = test_set[base_neighbors[i][2]][1]
        l3 = test_set[base_neighbors[i][3]][1]
        plt.title(CIFAR100_LABELS_LIST[l1] + "(" + str(lmapping[l1]) + ") " + CIFAR100_LABELS_LIST[l2] +
                  "(" + str(lmapping[l2]) + ") " + CIFAR100_LABELS_LIST[l3] + "(" + str(lmapping[l3]) + ")")
        plt.axis('off')
        plt.imshow(new_im)

    plt.show()
