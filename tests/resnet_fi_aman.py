
from pytorchfi.core import fault_injection 
from pytorchfi.neuron_error_models import random_neuron_inj
import torch.nn as nn
from collections import namedtuple
from pytorchfi import core
import torch
import os
import logging
from src.utils.helpers import get_activation
from data.data import testloader
from src.utils.helpers import device
from src.models.resnet import resnet18
from src.utils.customFI_methods import random_neuron_single_bit_inj_Aman
from src.utils.customFI_methods import single_bit_flip_func
import sys
from pytorchfi.neuron_error_models import (
    random_inj_per_layer,
    random_inj_per_layer_batched,
    random_neuron_inj,
    random_neuron_inj_batched,
    random_neuron_single_bit_inj,
    random_neuron_single_bit_inj_batched,
    random_batch_element,
    random_neuron_location,
    #declare_neuron_fault_injection
)
# os.environ['LOGLEVEL'] = 'DEBUG'  # Adjust logging level to capture DEBUG messages
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)-15s %(levelname)s %(message)s',
#                     handlers=[
#                         logging.StreamHandler(sys.stdout),  # Log to console
#                         logging.FileHandler('logfile.txt')  # Log to file
#                     ])


'''
custom bit flip + exact neuron target
'''
net = resnet18(pretrained=True, progress=True)
num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, 10)  # Modify the last layer to output 10 classes
net = net.to(device)


batch_size = 1

H = 32
W = 32
C = 3
bit_pos = 0
ranges = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
pfi = single_bit_flip_func(
            net,
            batch_size=batch_size,
            input_shape=[C,H,W],
            use_cuda=torch.cuda.is_available(),  # NOTE: was hardcoded True; crashed on any CPU-only machine
            bits=8,
            random_batch=False,
            bit_pos=bit_pos,
        )

fi_layer=3
fi_c = 33
fi_h = 8
fi_w = 7

corrupt_model = random_neuron_single_bit_inj_Aman(pfi, ranges, fi_layer, fi_c, fi_h, fi_w,bit_pos = bit_pos)


def test_with_fault(epoch):
    
    # b, layer, C, H, W, err_val = [0], [0], [0], [0], [0], [1000]
    test_loss = 0
    correct = 0
    total = 0
    count= 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            pfi.reset_current_layer()
            #single_input = inputs[0]  # Extract the first image in the batch
            #outputs = corrupt_model(single_input.unsqueeze(0))
            outputs = corrupt_model(inputs)
            
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            count += 1
            if count == 5 :
                print(len(outputs), "== batchsize")
    print(f'\nTest set: Average loss: {test_loss/len(testloader):.4f}, Accuracy: {correct}/{total} ({100.*correct/total:.2f}%)\n')

if __name__ == '__main__':
    test_with_fault(0)
