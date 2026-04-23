import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, spikegen
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import os

print("Started script")

class SurrogateSNN(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()

        # surrogate gradient for backprop through spikes
        spike_grad = surrogate.fast_sigmoid()

        # conv layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # pooling
        self.pool = nn.MaxPool2d(2, 2)

        # fc layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # LIF neurons
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif6 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, num_steps=25):
        # undo normalization for rate coding
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)

        # make spike trains over time
        spk_in = spikegen.rate(x, num_steps=num_steps)

        # init membrane states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()

        # store output spikes and membrane values
        spk_out_rec = []
        mem_out_rec = []
        # count spikes per layer across all time steps
        layer_spikes = [0] * 6
        layer_totals = [0] * 6
        # count approximate synaptic operations per layer
        layer_synops = [0] * 6

        for step in range(num_steps):
            # conv1 -> lif -> pool
            cur1 = self.conv1(spk_in[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            # count spikes in layer 1
            layer_spikes[0] += spk1.sum().item()
            layer_totals[0] += spk1.numel()

            x1 = self.pool(spk1)

            # synops from pooled layer 1 spikes into conv2
            layer_synops[0] += x1.sum().item() * (64 * 3 * 3)

            # conv2 -> lif -> pool
            cur2 = self.conv2(x1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # count spikes in layer 2
            layer_spikes[1] += spk2.sum().item()
            layer_totals[1] += spk2.numel()

            x2 = self.pool(spk2)

            # synops from pooled layer 2 spikes into conv3
            layer_synops[1] += x2.sum().item() * (128 * 3 * 3)

            # conv3 -> lif -> pool
            cur3 = self.conv3(x2)
            spk3, mem3 = self.lif3(cur3, mem3)

            # count spikes in layer 3
            layer_spikes[2] += spk3.sum().item()
            layer_totals[2] += spk3.numel()

            x3 = self.pool(spk3)

            # synops from pooled layer 3 spikes into fc1
            layer_synops[2] += x3.sum().item() * 256

            # flatten
            x3 = torch.flatten(x3, 1)

            # fc1 -> lif
            cur4 = self.fc1(x3)
            spk4, mem4 = self.lif4(cur4, mem4)

            # count spikes in layer 4
            layer_spikes[3] += spk4.sum().item()
            layer_totals[3] += spk4.numel()

            # synops from layer 4 spikes into fc2
            layer_synops[3] += spk4.sum().item() * 128

            # fc2 -> lif
            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)

            # count spikes in layer 5
            layer_spikes[4] += spk5.sum().item()
            layer_totals[4] += spk5.numel()

            # synops from layer 5 spikes into fc3
            layer_synops[4] += spk5.sum().item() * 10

            # fc3 -> output lif
            cur6 = self.fc3(spk5)
            spk6, mem6 = self.lif6(cur6, mem6)

            # count spikes in layer 6
            layer_spikes[5] += spk6.sum().item()
            layer_totals[5] += spk6.numel()
            # output layer has no next trainable layer, keep as 0
            layer_synops[5] += 0

            # save output over time
            spk_out_rec.append(spk6)
            mem_out_rec.append(mem6)

        # return outputs plus spike statistics
        return torch.stack(spk_out_rec), torch.stack(mem_out_rec), layer_spikes, layer_totals, layer_synops
    
print("Loading datasets...")

# preprocessing
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

# full training set with augmentation
train_dataset_full = torchvision.datasets.CIFAR10(
    root='./utils',
    train=True,
    transform=train_transform,
    download=True
)

# validation set without augmentation
val_dataset_full = torchvision.datasets.CIFAR10(
    root='./utils',
    train=True,
    transform=test_transform,
    download=True
)

# test set
test_data = torchvision.datasets.CIFAR10(
    root='./utils',
    train=False,
    transform=test_transform,
    download=True
)

# split indices
indices = np.random.RandomState(42).permutation(50000)
train_indices = indices[:45000]
val_indices = indices[45000:]

# create train/val subsets
train_data = Subset(train_dataset_full, train_indices)
val_data = Subset(val_dataset_full, val_indices)

# dataloaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

print("Dataloaders ready")

# number of time steps
T = 25

# create model
model = SurrogateSNN()

print("Model created")

# count trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model size (trainable parameters): {total_params}")

# training flag
TRAIN = True

# making sure folder exists
os.makedirs('./results/models', exist_ok=True)

# evaluate model at a specific number of time steps
def evaluate_model(model, loader, eval_T):
    model.eval()

    test_correct = 0
    test_total = 0

    # track spikes across full evaluation set
    test_layer_spikes = [0] * 6
    test_layer_totals = [0] * 6

    test_layer_synops = [0] * 6

    with torch.no_grad():
        for images, labels in loader:
            # forward pass with chosen T
            spk_out, mem_out, layer_spikes, layer_totals, layer_synops = model(images, num_steps=eval_T)

            # sum membrane potentials over time
            mem_sum = torch.sum(mem_out, dim=0)

            # predictions
            _, predicted = torch.max(mem_sum, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # accumulate layer stats
            for i in range(6):
                test_layer_spikes[i] += layer_spikes[i]
                test_layer_totals[i] += layer_totals[i]
                test_layer_synops[i] += layer_synops[i]

    # accuracy
    test_accuracy = 100 * test_correct / test_total

    # total spike count
    total_spike_count = sum(test_layer_spikes)

    # approximate total synops
    total_synops = sum(test_layer_synops)

    # average spikes per image
    avg_spikes_per_image = total_spike_count / test_total

    # firing rates
    firing_rates = [test_layer_spikes[i] / test_layer_totals[i] for i in range(6)]

    # sparsity = 1 - firing rate
    sparsities = [1.0 - firing_rates[i] for i in range(6)]

    # overall firing rate / sparsity
    overall_firing_rate = total_spike_count / sum(test_layer_totals)
    overall_sparsity = 1.0 - overall_firing_rate

    return {
        "T": eval_T,
        "accuracy": test_accuracy,
        "total_spike_count": total_spike_count,
        "total_synops": total_synops,
        "avg_spikes_per_image": avg_spikes_per_image,
        "firing_rates": firing_rates,
        "sparsities": sparsities,
        "overall_firing_rate": overall_firing_rate,
        "overall_sparsity": overall_sparsity
    }

if TRAIN:
    # loss on accumulated membrane potentials
    loss_function = nn.CrossEntropyLoss()

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # number of epochs
    num_epochs = 20

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1} ...')

        start = time.perf_counter()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # clear old gradients
            optimizer.zero_grad()

            # forward pass over all time steps
            spk_out, mem_out, _, _, _ = model(images, num_steps=T)

            # sum membrane potentials over time
            mem_sum = torch.sum(mem_out, dim=0)

            # compute classification loss
            loss = loss_function(mem_sum, labels)

            # backprop through all time steps
            loss.backward()

            # update weights
            optimizer.step()

            # track loss
            running_loss += loss.item()

            # get predictions
            _, predicted = torch.max(mem_sum, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # epoch stats
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")

        # validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                # forward pass
                spk_out, mem_out, _, _, _ = model(images, num_steps=T)

                # sum membrane potentials over time
                mem_sum = torch.sum(mem_out, dim=0)

                # predictions
                _, predicted = torch.max(mem_sum, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        print(f"Validation Accuracy = {val_accuracy:.2f}%")

        # save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), './results/models/best_surrogate_snn.pth')
            print("Best surrogate SNN saved.")

        end = time.perf_counter()
        total_seconds = end - start
        minutes, seconds = divmod(total_seconds, 60)
        print(f"Elapsed time: {int(minutes)}:{seconds:05.2f} minutes")

    # load best saved model before final evaluation
    model.load_state_dict(torch.load('./results/models/best_surrogate_snn.pth'))
    print("\nLoaded best surrogate SNN for final evaluation.")

    # evaluate at multiple time-step settings
    results_10 = evaluate_model(model, test_loader, eval_T=10)
    results_25 = evaluate_model(model, test_loader, eval_T=25)
    results_50 = evaluate_model(model, test_loader, eval_T=50)

    print("\n===== Efficiency Comparison Across Time Steps =====")

    for results in [results_10, results_25, results_50]:
        print(f"\nT = {results['T']}")
        print(f"Test Accuracy: {results['accuracy']:.2f}%")
        print(f"Total spike count: {results['total_spike_count']}")
        print(f"Approximate total SynOps: {results['total_synops']}")
        print(f"Average spikes per image: {results['avg_spikes_per_image']:.2f}")
        print(f"Overall firing rate: {results['overall_firing_rate']:.6f}")
        print(f"Overall sparsity: {results['overall_sparsity']:.6f}")

        print(f"Layer 1 firing rate: {results['firing_rates'][0]:.6f} | sparsity: {results['sparsities'][0]:.6f}")
        print(f"Layer 2 firing rate: {results['firing_rates'][1]:.6f} | sparsity: {results['sparsities'][1]:.6f}")
        print(f"Layer 3 firing rate: {results['firing_rates'][2]:.6f} | sparsity: {results['sparsities'][2]:.6f}")
        print(f"Layer 4 firing rate: {results['firing_rates'][3]:.6f} | sparsity: {results['sparsities'][3]:.6f}")
        print(f"Layer 5 firing rate: {results['firing_rates'][4]:.6f} | sparsity: {results['sparsities'][4]:.6f}")
        print(f"Layer 6 firing rate: {results['firing_rates'][5]:.6f} | sparsity: {results['sparsities'][5]:.6f}")

else:
    # load best saved model before final evaluation
    model.load_state_dict(torch.load('./results/models/best_surrogate_snn.pth'))
    print("\nLoaded best surrogate SNN for final evaluation.")

    # evaluate at multiple time-step settings
    results_10 = evaluate_model(model, test_loader, eval_T=10)
    results_25 = evaluate_model(model, test_loader, eval_T=25)
    results_50 = evaluate_model(model, test_loader, eval_T=50)

    print("\n===== Efficiency Comparison Across Time Steps =====")

    for results in [results_10, results_25, results_50]:
        print(f"\nT = {results['T']}")
        print(f"Test Accuracy: {results['accuracy']:.2f}%")
        print(f"Total spike count: {results['total_spike_count']}")
        print(f"Approximate total SynOps: {results['total_synops']}")
        print(f"Average spikes per image: {results['avg_spikes_per_image']:.2f}")
        print(f"Overall firing rate: {results['overall_firing_rate']:.6f}")
        print(f"Overall sparsity: {results['overall_sparsity']:.6f}")
        print(f"Overall sparsity (%): {results['overall_sparsity'] * 100:.2f}%")

        print(f"Layer 1 firing rate: {results['firing_rates'][0]:.6f} | sparsity: {results['sparsities'][0]:.6f}")
        print(f"Layer 2 firing rate: {results['firing_rates'][1]:.6f} | sparsity: {results['sparsities'][1]:.6f}")
        print(f"Layer 3 firing rate: {results['firing_rates'][2]:.6f} | sparsity: {results['sparsities'][2]:.6f}")
        print(f"Layer 4 firing rate: {results['firing_rates'][3]:.6f} | sparsity: {results['sparsities'][3]:.6f}")
        print(f"Layer 5 firing rate: {results['firing_rates'][4]:.6f} | sparsity: {results['sparsities'][4]:.6f}")
        print(f"Layer 6 firing rate: {results['firing_rates'][5]:.6f} | sparsity: {results['sparsities'][5]:.6f}")