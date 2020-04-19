import json

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image


if __name__ == '__main__':
    neighbors = np.load('../Model/nn_experiment.npy')
    test_set = torchvision.datasets.CIFAR100(root='./cifar100', train=False, download=True)

    with open('../Model/specs.jsons', 'r') as f:
        specialties = json.load(f)

    for i in range(4, 10):
        plt.subplot(1, 2, 1)
        plt.imshow(test_set[i][0])

        images = [test_set[neighbors[i][1]][0], test_set[neighbors[i][2]][0], test_set[neighbors[i][3]][0]]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        plt.subplot(1, 2, 2)
        plt.imshow(new_im)
        plt.show()
