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

