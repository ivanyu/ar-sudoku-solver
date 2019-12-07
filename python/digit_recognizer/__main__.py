#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from .model import DigitRecognizer
from .mnist_no_zeros import MNISTNoZero
from .typed_digits_dataset import TypedDigits


DIGITS_DATASET_FOLDER = "../digits_dataset"
MODEL_FILE = "model.pth"
MODEL_FINETUNED_FILE = "model-ft.pth"

DEVICE = torch.device("cuda:0")
# DEVICE = torch.device("cpu")

TRANSFORM_TO_TENSOR = tv.transforms.ToTensor()
TRANSFORM_NORMALIZER = tv.transforms.Normalize((0.5,), (0.5,))


def validate(model, validation_loader):
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
        correct += (torch.max(outputs.data, dim=1)[1] == labels).float().sum()
        total += labels.shape[0]
        loss += criterion(outputs, labels).item()
        batch_count += 1
    if was_in_train_mode:
        model.train()
    return (loss / batch_count, correct / total)


def train(model, train_loader, validation_loader, epochs):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(trainable_params, lr=0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss_train = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item()
            if i % 200 == 199:
                if validation_loader is not None:
                    val_loss, val_accuracy = validate(model, validation_loader)
                    message = (f"[{epoch + 1}, {i + 1:5d}] train loss: {running_loss_train / 200:.3f}\t" +
                               f"val loss: {val_loss:.3f}\t" +
                               f"val accuracy: {val_accuracy:.3f}%")
                    print(message)
                else:
                    print(f"[{epoch + 1}, {i + 1:5d}] train loss: {running_loss_train / 200:.3f}")
                running_loss_train = 0.0
    pass


def test(model):
    transform = tv.transforms.Compose([
        tv.transforms.Grayscale(),
        TRANSFORM_TO_TENSOR,
        TRANSFORM_NORMALIZER
    ])
    test_data = tv.datasets.ImageFolder(DIGITS_DATASET_FOLDER, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32,
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
    if was_in_train_mode:
        model.train()
    print(f"Test acc: {correct / total:.3f}%")


pretrained = os.path.exists(MODEL_FILE)

model = DigitRecognizer()
if pretrained:
    model.load_state_dict(torch.load(MODEL_FILE))
model.to(DEVICE)

if not pretrained:
    transform_augment = tv.transforms.Compose([
        tv.transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), shear=10),
        TRANSFORM_TO_TENSOR,
        TRANSFORM_NORMALIZER
    ])
    train_data = MNISTNoZero(root='./data', train=True,
                             download=True, transform=transform_augment)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
                                               shuffle=True, num_workers=0)

    transform = tv.transforms.Compose([
        TRANSFORM_TO_TENSOR,
        TRANSFORM_NORMALIZER
    ])
    validation_data = MNISTNoZero(root='./data', train=False,
                                  download=True, transform=transform)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32,
                                                    shuffle=False, num_workers=0)

    print("Training")
    train(model, train_loader, validation_loader, 7)
    print("Finished training")
    torch.save(model.state_dict(), MODEL_FILE)

test(model)

transform_augment = tv.transforms.Compose([
    tv.transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), shear=10),
    TRANSFORM_TO_TENSOR,
    TRANSFORM_NORMALIZER
])
train_data = TypedDigits(transform=transform_augment)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
                                           shuffle=True, num_workers=0)

print("Fine tuning")
model.freeze()
train(model, train_loader, None, 3)
print("Finished fine tuning")
test(model)
torch.save(model.state_dict(), MODEL_FINETUNED_FILE)
