# -*- coding: utf-8 -*-
import torch
import torchvision as tv


class MNISTNoZero(tv.datasets.MNIST):
    # classes = ['1 - one', '2 - two', '3 - three', '4 - four',
    #            '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNISTNoZero, self).__init__(root, train=train,
                                          transform=transform,
                                          target_transform=target_transform,
                                          download=download)

        t = self.targets
        self.data = self.data[t != 0]
        self.targets = self.targets[t != 0] - 1

    # @property
    # def class_to_idx(self):
    #     return {_class: i - 1 for i, _class in enumerate(self.classes)}


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    transform_augment = tv.transforms.Compose([
        tv.transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), shear=10),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = MNISTNoZero(root='./data', train=True,
                          download=True, transform=transform_augment)
    loader = torch.utils.data.DataLoader(dataset, batch_size=40,
                                         shuffle=True, num_workers=0)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    imshow(tv.utils.make_grid(images))
