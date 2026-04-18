import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset


class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

if __name__ == "__main__":

    # Preprocessing
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset_full = torchvision.datasets.CIFAR10(
        root='./utils',
        train=True,
        transform=train_transform,
        download=True
    )

    val_dataset_full = torchvision.datasets.CIFAR10(
        root='./utils',
        train=True,
        transform=test_transform,
        download=True
    )

    test_data = torchvision.datasets.CIFAR10(
        root='./utils',
        train=False,
        transform=test_transform,
        download=True
    )

    indices = np.random.RandomState(42).permutation(50000)
    train_indices = indices[:45000]
    val_indices = indices[45000:]

    train_data = Subset(train_dataset_full, train_indices)
    val_data = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

    image, label = train_dataset_full[0]

    print("\nImage Size in Dataset:")
    print(image.size())

    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # create instance
    net = NeuralNet()
    # determine loss function method
    loss_function = nn.CrossEntropyLoss()
    # determine learning rules
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    # Results
    os.makedirs('./results/models', exist_ok=True)
    os.makedirs('./results/plots', exist_ok=True)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0

    # Training flag
    TRAIN = False

    if TRAIN:
        for epoch in range(20):
            print(f'\nTraining epoch {epoch + 1}...')

            net.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels in train_loader:
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = loss_function(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_accuracy = 100 * correct_train / total_train

            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_accuracy)

            # validation
            net.eval()
            val_running_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = net(inputs)
                    loss = loss_function(outputs, labels)

                    val_running_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            epoch_val_loss = val_running_loss / len(val_loader)
            epoch_val_accuracy = 100 * correct_val / total_val

            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_accuracy)

            print(f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_accuracy:.2f}%')
            print(f'Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_accuracy:.2f}%')

            if epoch_val_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_val_accuracy
                torch.save(net.state_dict(), './results/models/best_net.pth')
                print('Best model saved.')

    else:
        net.load_state_dict(torch.load('./results/models/best_net.pth'))

    # always load best model before final test evaluation
    net.load_state_dict(torch.load('./results/models/best_net.pth'))

    #load net in another file
    #net = NeuralNet()
    #net.load_state_dict(torch.load('./results/models/trained_net.pth))

    correct = 0
    total = 0

    net.eval()

    start_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time = time.perf_counter()

    test_accuracy = 100 * correct / total
    total_inference_time = end_time - start_time
    ms_per_image = (total_inference_time / total) * 1000

    print(f'\nFinal Test Accuracy: {test_accuracy:.2f}%')
    print(f'Average Inference Time: {ms_per_image:.4f} ms/image')


    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('./results/plots/loss_curve.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('./results/plots/accuracy_curve.png')
    plt.close()

    # new_transform = transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    # def load_image(image_path):
    #     image = Image.open(image_path)
    #     image = new_transform(image)
    #     image = image.unsqueeze(0)
    #     return image

    #image_paths = []
    #images = [load_image(img) for img in image_paths]

    #net.eval()
    # with torch.no_grad():
    #     for image in images:
    #         output = net(image)
    #         _, predicted = torch.max(output, 1)
    #         print(f'Prediction: {class_names[predicted.item()]}')