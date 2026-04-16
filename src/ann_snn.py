from cnn import NeuralNet
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
import sys

# Create CNN instance
cnn = NeuralNet()
# Load trained CNN
cnn.load_state_dict(torch.load('./results/models/best_net.pth'))

# Check if Loading CNN was successful
if cnn.eval():
    print('\nArchitecture loaded successfully.')
else:
    print('\nError loading Architecture.')
    sys.exit()

print('\nCNN weights:')
if cnn.conv1.weight is not None:
    print(f'\nConv 1 Weights: Available')
else:
    ('No Weights')
if cnn.conv1.bias is not None:
    print(f'Conv 1 Bias: Available')
else:
    print('No Bias')
if cnn.conv2.weight is not None:
    print(f'Conv 2 Weights: Available')
else:
    print('No Weights')
if cnn.conv2.bias is not None:
    print(f'Conv 2 Bias: Available')
else:
    print('No Bias')
if cnn.conv3.weight is not None:
    print(f'Conv 3 Weights: Available')
else:
    print('No Weights')
if cnn.conv3.bias is not None:
    print(f'Conv 3 Bias: Available')
else:
    print('No Bias')
if cnn.fc1.weight is not None:
    print(f'FC 1 Weights: Available')
else:
    print('No Bias')
if cnn.fc2.weight is not None:
    print(f'FC 2 Weights: Available')
else:
    print('No FC')
if cnn.fc3.weight is not None:
    print(f'FC 3 Wieghts: Available')
else:
    print('No FC')


print(f'\nConv 1 Shape: {cnn.conv1.weight.shape}')
print(f'Conv 2 Shape: {cnn.conv2.weight.shape}')
print(f'Conv 3 Shape: {cnn.conv3.weight.shape}')
print(f'\nFC 1 Shape: {cnn.fc1.weight.shape}')
print(f'FC 2 Shape: {cnn.fc2.weight.shape}')
print(f'FC 3 Shape: {cnn.fc3.weight.shape}')

print('\n---------------- Converting into SNN ----------------')

class ConvertedSNN(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()

        # same structure as CNN
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # spiking neurons (LIF)
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)
        self.lif3 = snn.Leaky(beta=beta)
        self.lif4 = snn.Leaky(beta=beta)
        self.lif5 = snn.Leaky(beta=beta)
        self.lif6 = snn.Leaky(beta=beta)

    def forward(self, x, num_steps=50):

        # convert input to spikes
        spk_in = spikegen.rate(x, num_steps=num_steps)

        # initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()

        # keep track of spikes
        spk_out_rec = []


        for step in range(num_steps):
            # applies convolution to spike input at this time step -> produces current
            cur1 = self.conv1(spk_in[step])
            # uses produced spike and previous membrane potential to update the membrane and output binary spike
            spk1, mem1 = self.lif1(cur1, mem1)
            # pools spike feature map
            x1 = self.pool(spk1)

            cur2 = self.conv2(x1)
            spk2, mem2 = self.lif2(cur2, mem2)
            x2 = self.pool(spk2)

            cur3 = self.conv3(x2)
            spk3, mem3 = self.lif3(cur3, mem3)
            x3 = self.pool(spk3)

            # flatten feature map for fc
            x3 = torch.flatten(x3, 1)

            # passes flattened feature vector through fc layer -> produces current
            cur4 = self.fc1(x3)
            # lif neuron uses produced spike and previous membrane potential to update the membrane and output binary spike
            spk4, mem4 = self.lif4(cur4, mem4)

            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)

            cur6 = self.fc3(spk5)
            spk6, mem6 = self.lif6(cur6, mem6)

            # tensor of spikes
            spk_out_rec.append(spk6)

        # convert all tensors into 1 single tensor and return
        return torch.stack(spk_out_rec)

snn = ConvertedSNN()
print('\nSNN created successfully.')