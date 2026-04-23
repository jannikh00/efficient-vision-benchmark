from cnn import NeuralNet
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

# Create CNN instance
cnn = NeuralNet()
# Load trained CNN
cnn.load_state_dict(torch.load('./results/models/best_net.pth'))

# switch CNN to evaluation mode
cnn.eval()
print('\nArchitecture loaded successfully.')

print('\n---------------- Converting into SNN ----------------')

# number of time steps for temporal inference
T = 50

class ConvertedSNN(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()

        # same conv structure as CNN
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # use max pooling to match original CNN
        self.pool = nn.MaxPool2d(2, 2)

        # keep batch norm because original CNN was trained with it
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # same fully connected structure as CNN
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # spiking neurons
        self.lif1 = snn.Leaky(beta=beta, threshold=0.5)
        self.lif2 = snn.Leaky(beta=beta, threshold=0.5)
        self.lif3 = snn.Leaky(beta=beta, threshold=0.5)
        self.lif4 = snn.Leaky(beta=beta, threshold=0.5)
        self.lif5 = snn.Leaky(beta=beta, threshold=0.5)
        self.lif6 = snn.Leaky(beta=beta, threshold=0.5)

    def forward(self, x, num_steps=T):
        # undo normalization: [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0

        # make sure values stay in valid spike probability range
        x = torch.clamp(x, 0.0, 1.0)

        # rate coding: create spikes over time
        spk_in = spikegen.rate(x, num_steps=num_steps)

        # initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()

        # store output spikes of final layer for each time step
        spk_out_rec = []

        # count total spikes across all layers
        # count total spikes across all layers
        total_spikes = 0

        # count total possible spikes across all layers
        total_neurons = 0

        # count approximate synaptic operations per layer
        layer_synops = [0] * 6

        for step in range(num_steps):
            # conv -> batch norm -> LIF -> pool
            cur1 = self.bn1(self.conv1(spk_in[step]))
            spk1, mem1 = self.lif1(cur1, mem1)
            x1 = self.pool(spk1)

            # synops from pooled layer 1 spikes into conv2
            layer_synops[0] += x1.sum().item() * (64 * 3 * 3)

            cur2 = self.bn2(self.conv2(x1))
            spk2, mem2 = self.lif2(cur2, mem2)
            x2 = self.pool(spk2)

            # synops from pooled layer 2 spikes into conv3
            layer_synops[1] += x2.sum().item() * (128 * 3 * 3)

            cur3 = self.bn3(self.conv3(x2))
            spk3, mem3 = self.lif3(cur3, mem3)
            x3 = self.pool(spk3)

            # synops from pooled layer 3 spikes into fc1
            layer_synops[2] += x3.sum().item() * 256

            # flatten for fully connected layers
            x3 = torch.flatten(x3, 1)

            cur4 = self.fc1(x3)
            spk4, mem4 = self.lif4(cur4, mem4)

            # synops from layer 4 spikes into fc2
            layer_synops[3] += spk4.sum().item() * 128

            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)

            # synops from layer 5 spikes into fc3
            layer_synops[4] += spk5.sum().item() * 10

            cur6 = self.fc3(spk5)
            spk6, mem6 = self.lif6(cur6, mem6)

            # output layer has no next trainable layer
            layer_synops[5] += 0

            # save final output spikes
            spk_out_rec.append(spk6)

            # add spikes from all layers
            total_spikes += (
                spk1.sum() + spk2.sum() + spk3.sum() +
                spk4.sum() + spk5.sum() + spk6.sum()
            ).item()

            # count all possible spike positions
            total_neurons += (
                spk1.numel() + spk2.numel() + spk3.numel() +
                spk4.numel() + spk5.numel() + spk6.numel()
            )

        # return:
        # 1) output spikes over time
        # 2) total spike count
        # 3) total possible spike positions
        # 4) approximate synaptic operations
        return torch.stack(spk_out_rec), total_spikes, total_neurons, layer_synops

# create SNN instance
snn = ConvertedSNN()
print('\nSNN created successfully.')

# copy weights
snn.conv1.weight.data = cnn.conv1.weight.data.clone()
snn.conv2.weight.data = cnn.conv2.weight.data.clone()
snn.conv3.weight.data = cnn.conv3.weight.data.clone()
snn.fc1.weight.data = cnn.fc1.weight.data.clone()
snn.fc2.weight.data = cnn.fc2.weight.data.clone()
snn.fc3.weight.data = cnn.fc3.weight.data.clone()

print ('\nWeights copied successfully.')

# copy biases

snn.conv1.bias.data = cnn.conv1.bias.data.clone()
snn.conv2.bias.data = cnn.conv2.bias.data.clone()
snn.conv3.bias.data = cnn.conv3.bias.data.clone()
snn.fc1.bias.data = cnn.fc1.bias.data.clone()
snn.fc2.bias.data = cnn.fc2.bias.data.clone()
snn.fc3.bias.data = cnn.fc3.bias.data.clone()

print ('\nBiases copied successfully.')

# copy batch norm learnable parameters
snn.bn1.weight.data = cnn.bn1.weight.data.clone()
snn.bn1.bias.data = cnn.bn1.bias.data.clone()
snn.bn2.weight.data = cnn.bn2.weight.data.clone()
snn.bn2.bias.data = cnn.bn2.bias.data.clone()
snn.bn3.weight.data = cnn.bn3.weight.data.clone()
snn.bn3.bias.data = cnn.bn3.bias.data.clone()

# copy batch norm running statistics
snn.bn1.running_mean.data = cnn.bn1.running_mean.data.clone()
snn.bn1.running_var.data = cnn.bn1.running_var.data.clone()
snn.bn2.running_mean.data = cnn.bn2.running_mean.data.clone()
snn.bn2.running_var.data = cnn.bn2.running_var.data.clone()
snn.bn3.running_mean.data = cnn.bn3.running_mean.data.clone()
snn.bn3.running_var.data = cnn.bn3.running_var.data.clone()

print('\nBatchNorm parameters copied successfully.')

print('\n---------------- Testing ----------------')

# same normalization as in CNN
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# same CIFAR-10 test set
test_data = torchvision.datasets.CIFAR10(
    root='./utils',
    train=False,
    transform=test_transform,
    download=True
)

# same batch size
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

# number of correct predictions
correct = 0

# total number of images
total = 0

# total spikes across whole test set
dataset_total_spikes = 0

# total possible spikes across whole test set
dataset_total_neurons = 0

# total approximate synaptic operations across whole test set
dataset_total_synops = 0

# sum of stable prediction steps across all images
stable_step_sum = 0

# put model in evaluation mode
snn.eval()

# no gradient computation
with torch.no_grad():
    # load batches of images and true labels
    for i, (images, labels) in enumerate(test_loader):
        if i % 50 == 0:
            print(f"\nProcessing batch {i}")

        # run SNN for T time steps
        outputs, batch_spikes, batch_neurons, batch_synops = snn(images, num_steps=T)

        # sum output spikes over time
        summed = torch.sum(outputs, dim=0)

        # final prediction = class with most cumulative spikes
        _, predicted = torch.max(summed, 1)

        # update accuracy counters
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # update spike counters
        dataset_total_spikes += batch_spikes
        dataset_total_neurons += batch_neurons
        dataset_total_synops += sum(batch_synops)

        # find stable prediction step for each image
        cumulative = torch.zeros_like(summed)
        batch_size = labels.size(0)

        for sample_idx in range(batch_size):
            stable_step = T

            for step in range(T):
                # cumulative spikes up to this time step
                current_sum = torch.sum(outputs[:step + 1, sample_idx, :], dim=0)
                current_pred = torch.argmax(current_sum).item()

                # final prediction for this sample
                final_pred = predicted[sample_idx].item()

                # if current prediction already matches final prediction,
                # check whether it stays the same for all later steps
                if current_pred == final_pred:
                    stays_same = True

                    for future_step in range(step, T):
                        future_sum = torch.sum(outputs[:future_step + 1, sample_idx, :], dim=0)
                        future_pred = torch.argmax(future_sum).item()

                        if future_pred != final_pred:
                            stays_same = False
                            break

                    if stays_same:
                        stable_step = step + 1
                        break

            stable_step_sum += stable_step

# compute final accuracy
accuracy = 100 * correct / total

# average time step until prediction becomes stable
avg_stable_steps = stable_step_sum / total

# average spikes per image
avg_spikes_per_inference = dataset_total_spikes / total

# average firing rate
avg_firing_rate = dataset_total_spikes / dataset_total_neurons

# overall / global sparsity
avg_sparsity = 1.0 - avg_firing_rate

# print final stats
print(f"\nTotal samples: {total}")
print(f"Correct predictions: {correct}")
print(f"SNN Test Accuracy: {accuracy:.2f}%")
print(f"Chosen time steps T: {T}")
print(f"Average stable prediction step: {avg_stable_steps:.2f}")
print(f"Average spikes per inference: {avg_spikes_per_inference:.2f}")
print(f"Average firing rate: {avg_firing_rate:.6f}")
print(f"Average sparsity: {avg_sparsity:.6f}")
print(f"Average sparsity (%): {avg_sparsity * 100:.2f}%")
print(f"Approximate total SynOps: {dataset_total_synops}")
print(f"Approximate SynOps per inference: {dataset_total_synops / total:.2f}")