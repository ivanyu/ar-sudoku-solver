#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import string
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from torch.utils.data import Subset
import matplotlib.pyplot as plt

from .augmentation import draw_borders, gaussian_noise, gaussian_blur, salt_and_pepper_noise, scratches, scale_and_shift
from .loss_reporter import LossReporter
from .model import DigitRecognizer2


DEVICE = torch.device("cuda:0")
# DEVICE = torch.device("cpu")

TRANSFORM_GRAYSCALE = tv.transforms.Grayscale()
TRANSFORM_TO_TENSOR = tv.transforms.ToTensor()
TRANSFORM_NORMALIZER = tv.transforms.Normalize((0.5,), (0.5,))

MODEL_FILE = "model-ft.pth"


def validate(model, validation_loader, save_misclassified):
    correct = 0.0
    total = 0.0
    loss = 0.0
    batch_count = 0
    criterion = nn.CrossEntropyLoss()

    was_in_train_mode = model.training
    model.eval()
    for i, data in enumerate(validation_loader, 0):
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(inputs)

        if save_misclassified:
            incorrect_indxs = torch.where(torch.max(outputs.data, dim=1)[1] != labels)[0]
            os.makedirs("_misclassified", exist_ok=True)
            for incorrect_i in incorrect_indxs:
                input = tv.transforms.ToPILImage()(inputs[incorrect_i].cpu().squeeze())
                label = torch.max(outputs.data, dim=1)[1][incorrect_i].cpu().numpy().item() + 1
                # print(input, label)
                name = ''.join(random.choice(string.ascii_lowercase) for i in range(9))
                input.save(f"_misclassified/{name}-{label}.jpg")

        correct += (torch.max(outputs.data, dim=1)[1] == labels).float().sum()
        total += labels.shape[0]
        loss += criterion(outputs, labels).item()
        batch_count += 1
    if was_in_train_mode:
        model.train()
    return loss / batch_count, correct / total


def train(model, train_loader, validation_loader, epochs, plot_callback):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optim.AdamW(trainable_params, lr=0.001)
    # optimizer = optim.SGD(trainable_params, lr=0.001, momentum=0.9)
    optimizer = optim.SGD(trainable_params, lr=0.01, momentum=0.9, weight_decay=0.01)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 70, 100, 130, 150], gamma=0.3)

    criterion = nn.CrossEntropyLoss()
    train_loss = None
    val_loss = None
    val_accuracy = None

    loss_train_sum = 0.0
    batch_count = 0
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # import os
            # os.makedirs("x", exist_ok=True)
            # for i in range(inputs.shape[0]):
            #     img = tv.transforms.ToPILImage()(inputs[i, :, :, :].squeeze())
            #     import random
            #     import string
            #     name = ''.join(random.choice(string.ascii_lowercase) for i in range(9))
            #     img.save(f"x/{name}.jpg")

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            loss_train_sum += loss.item()
            batch_count += 1
            # show_every = 2
            # if i % show_every == show_every - 1:
            #     print(f"[epoch {epoch + 1}, {i + 1:2d}] train loss: {running_loss_train / show_every:.4f}")
            #     running_loss_train = 0.0
        if epoch % 10 == 0:
            if validation_loader is not None:
                val_loss, val_accuracy = validate(model, validation_loader, False)
                message = (f"[epoch {epoch + 1}]\t" +
                           f"train loss: {loss_train_sum / batch_count:.4f}\t" +
                           f"val loss: {val_loss:.4f}\t" +
                           f"val accuracy: {val_accuracy:.3f}%")
                print(message)
                plot_callback(epoch, train_loss, val_loss)

                loss_train_sum = 0.0
                batch_count = 0
            else:
                message = (f"[epoch {epoch + 1}]\t" +
                           f"train loss: {loss_train_sum / batch_count:.4f}")
                print(message)
                plot_callback(epoch, train_loss, None)

                loss_train_sum = 0.0
                batch_count = 0
    scheduler.step(epoch)
    return val_loss, val_accuracy


def test(model: nn.Module, save_misclassified):
    transform = tv.transforms.Compose([
        TRANSFORM_GRAYSCALE,
        TRANSFORM_TO_TENSOR,
        TRANSFORM_NORMALIZER
    ])

    real_digits_dataset = tv.datasets.ImageFolder("data/real_digits/dataset", transform=transform)
    test_loader = torch.utils.data.DataLoader(real_digits_dataset, batch_size=1000,
                                              shuffle=True, num_workers=0)
    correct = 0.0
    total = 0.0
    was_in_train_mode = model.training
    model.eval()
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(inputs).data
        correct += (torch.max(outputs, dim=1)[1] == labels).float().sum()
        total += labels.shape[0]

        if save_misclassified:
            incorrect_indxs = torch.where(torch.max(outputs.data, dim=1)[1] != labels)[0]
            os.makedirs("_misclassified", exist_ok=True)
            for incorrect_i in incorrect_indxs:
                input = tv.transforms.ToPILImage()(inputs[incorrect_i].cpu().squeeze())
                label = torch.max(outputs.data, dim=1)[1][incorrect_i].cpu().numpy().item() + 1
                # print(input, label)
                name = ''.join(random.choice(string.ascii_lowercase) for i in range(9))
                input.save(f"_misclassified/{name}-{label}.jpg")

    if was_in_train_mode:
        model.train()
    print(f"Test acc: {100.0 * correct / total:.3f}%")


transform_augment = tv.transforms.Compose([
    TRANSFORM_GRAYSCALE,
    scale_and_shift,
    gaussian_noise(mean=0, sigma=(0, 1)),
    tv.transforms.RandomApply([scratches()], 0.25),
    tv.transforms.RandomApply([salt_and_pepper_noise()], 0.3),
    TRANSFORM_TO_TENSOR,
    TRANSFORM_NORMALIZER
])
generated_digits_dataset = tv.datasets.ImageFolder("data/generated_digits", transform=transform_augment)


# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
# loader = torch.utils.data.DataLoader(generated_digits_dataset, batch_size=56,
#                                      shuffle=True, num_workers=0)
# dataiter = iter(loader)
# images1, _ = dataiter.next()
# dataiter = iter(loader)
# images2, _ = dataiter.next()
# images = torch.cat([images1, images2])
# imshow(tv.utils.make_grid(images))
# exit()


# print("Cross-validation")
# all_idxs = set(range(len(generated_digits_dataset)))
# folds = 10
# for current_fold in range(folds):
#     i = 0
#     validation_idxs = []
#     while (i * folds) + current_fold < len(generated_digits_dataset):
#         validation_idxs.append((i * folds) + current_fold)
#         i += 1
#     train_idxs = list(set(all_idxs) - set(validation_idxs))
#     train_idxs = sorted(train_idxs)
#     # print(validation_idxs, train_idxs)
#
#     train_set = Subset(generated_digits_dataset, train_idxs)
#     validation_set = Subset(generated_digits_dataset, validation_idxs)
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
#                                                shuffle=True, num_workers=0)
#     validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1000,
#                                                     shuffle=True, num_workers=0)
#
#     model: DigitRecognizer2 = DigitRecognizer2()
#     model.to(DEVICE)
#
#     epochs = 200
#     loss_reporter = LossReporter(epochs)
#     val_loss, val_accuracy = train(model, train_loader, validation_loader, epochs, loss_reporter)
#     break
# print("Finished cross-validation")
# exit()

model: DigitRecognizer2 = DigitRecognizer2()
model.to(DEVICE)
print("Training")
train_loader = torch.utils.data.DataLoader(generated_digits_dataset, batch_size=32,
                                           shuffle=True, num_workers=0)
epochs = 200
loss_reporter = LossReporter(epochs, report_validation=False)
train(model, train_loader, None, epochs, loss_reporter)
print("Finished training")
print("Testing")
test(model, save_misclassified=True)
print("Finished testing")
torch.save(model.state_dict(), MODEL_FILE)


# model.load_state_dict(torch.load(MODEL_FILE))
# model.eval()
# test(model, save_misclassified=True)
#
# img = Image.open("data/xx/7-8.jpg")
# tensor = TRANSFORM_TO_TENSOR(img).unsqueeze(1).float().to(DEVICE)
# outputs = model(tensor)
# print(generated_digits_dataset.class_to_idx)
# print(torch.max(outputs.data, dim=1)[1] + 1)
