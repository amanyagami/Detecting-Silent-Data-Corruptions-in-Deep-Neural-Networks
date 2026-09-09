

import torch.nn as nn

import torch

import copy
import csv
import pickle
import pandas as pd
# NOTE: removed `from data.data import testloader` -- that module doesn't exist
# in this repo and testloader is unused (shadowed below by a DataLoader built
# from correct_data.pt).
from src.utils.helpers import device
from torch.utils.data import TensorDataset, DataLoader
from src.models.resnet import resnet18
from src.utils.customFI_methods import random_neuron_single_bit_inj_Aman
from src.utils.customFI_methods import single_bit_flip_func
from config import Lambda2,Lambda1,Lambda3,bins_num,cohortSize,path
from scipy.stats import wasserstein_distance
# from offlineProfiling import cohort_size
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
cohort_size = cohortSize
LayerNames = []
from config import update_histogram,reset_histogram,normalize_histogram,listtohistogram,TAU2processing,TAU3processing,abnormality_score2,abnormality_score3,abnormility_score1

pathtoTau1 = path + "/tau1.pkl"
pathtoTau2 = path + "/tau2.pkl"
pathtoTau3 = path + "/tau3.pkl"
pathtoOutput = path + "/testwithoutSDC.csv"
pathtoDetectionSites = path + "/DetectionSites.txt"
activations= {}
def get_activation(name):
    def hook(model,input,output):
        activations[name] = output.detach()
    return hook

def test_without_fault(model, tau1, tau2, tau3):
    '''
    tau1 -> profiled DNA of neurons  
    TAU1 -> Layer wise (abnormality score of neurons added )
    tau2 -> layer DNA from profiling 
    TAU2 -> layer DNA during inference -> then abnormality score layer wise 
    tau3 -> profiled min max of layer  (need to take from profiling)
    TAU3 -> min max of a layer  -> the layer wise abnormality score

    '''
    TAU1 = {}
    Temp2 = copy.deepcopy(tau2)
    for layer_name in Temp2:
        Temp2[layer_name] =reset_histogram(Temp2[layer_name])
    TAU3 = {}
    model.eval()
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.bn1.register_forward_hook(get_activation('bn1'))
    model.layer1[0].conv1.register_forward_hook(get_activation('layer1.0.conv1'))
    model.layer1[0].bn1.register_forward_hook(get_activation('layer1.0.bn1'))
    model.layer1[0].conv2.register_forward_hook(get_activation('layer1.0.conv2'))
    model.layer1[0].bn2.register_forward_hook(get_activation('layer1.0.bn2'))
    model.layer2[0].conv1.register_forward_hook(get_activation('layer2.0.conv1'))
    model.layer2[0].bn1.register_forward_hook(get_activation('layer2.0.bn1'))
    model.layer2[0].conv2.register_forward_hook(get_activation('layer2.0.conv2'))
    model.layer2[0].bn2.register_forward_hook(get_activation('layer2.0.bn2'))
    model.layer3[0].conv1.register_forward_hook(get_activation('layer3.0.conv1'))
    model.layer3[0].bn1.register_forward_hook(get_activation('layer3.0.bn1'))
    model.layer3[0].conv2.register_forward_hook(get_activation('layer3.0.conv2'))
    model.layer3[0].bn2.register_forward_hook(get_activation('layer3.0.bn2'))
    model.layer4[0].conv1.register_forward_hook(get_activation('layer4.0.conv1'))
    model.layer4[0].bn1.register_forward_hook(get_activation('layer4.0.bn1'))
    model.layer4[0].conv2.register_forward_hook(get_activation('layer4.0.conv2'))
    model.layer4[0].bn2.register_forward_hook(get_activation('layer4.0.bn2'))
    model.avgpool.register_forward_hook(get_activation('avgpool'))
    model.fc.register_forward_hook(get_activation('fc'))

    # b, layer, C, H, W, err_val = [0], [0], [0], [0], [0], [1000]
    test_loss = 0
    correct = 0
    total = 0
    count= 0
    wrong = 0
    criterion = nn.CrossEntropyLoss()
    final_Scores = []
    # ignore_input= {} # to find out the inputs which are already producing wrong results.
    with open(pathtoOutput, 'w') as file:
        pass
    TAU1List = {}
    TAU2List = {}
    TAU3List = {}
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # pfi.reset_current_layer()
            #single_input = inputs[0]  # Extract the first image in the batch
            #outputs = corrupt_model(single_input.unsqueeze(0))
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            if predicted.eq(targets).sum().item()  != 1 :
                wrong +=1 
                continue
            else :
                correct += 1
            total += targets.size(0)
            count += 1
            
            
            TAU2 = copy.deepcopy(Temp2)

            for layer_name in LayerNames:
                TAU1[layer_name] = 0
            
            # TAU1
            for neuron in selected_neurons:
                layer_name,pos = neuron
                #some layers dont have 4 dimenions so this check (eg Fc ->last layer resnet18)
                if len(pos) > 2:       
                    b,c,w,h=pos
                    TAU1[layer_name] += abnormility_score1(activations[layer_name][b,c,w,h].item(),tau1[neuron])
                else :
                    w,h=pos
                    TAU1[layer_name] += abnormility_score1(activations[layer_name][b,c,w,h].item(),tau1[neuron])

            #TAU2
        
            
            # print(Temp2['avgpool'])
            for neuron in selected_neurons:
                layer_name,pos = neuron
            #some layers dont have 4 dimenions so this check (eg Fc ->last layer resnet18)
                if len(pos) > 2:       
                    b,c,w,h=pos
                    TAU2[layer_name] = update_histogram(activations[layer_name][b,c,w,h].item(),TAU2[layer_name])
                else :
                    w,h=pos
                    TAU2[layer_name] = update_histogram(activations[layer_name][w,h].item(),TAU2[layer_name])

            for layer_name in TAU2 :
                TAU2[layer_name] = abnormality_score2(TAU2[layer_name],tau2[layer_name])


            #TAU3
            for layer_name, tensor in activations.items():
                    TAU3[layer_name] = activations[layer_name]
            TAU3 = TAU3processing(TAU3,1)

            for layer_name in TAU3:
                TAU3[layer_name] = abnormality_score3(TAU3[layer_name],tau3[layer_name])

            #total abnormality score
            lambda1 = 1
            lambda2 = 1
            lambda3 = 1
            Total_score= {}
            prev = 0 
            #  generate tau values layer wise for each layer and write in the format given below
            '''

            row1 = (layer1, tau1[layer1],tau2[layer1],tau3[layer1]) , (layer2 , tau1[layer2], tau2[layer2],tau3[layer2]) ......
            row2 =  same as above but for second image/input 

            '''
            
            with open(pathtoOutput, mode='a', newline='') as file:
                writer = csv.writer(file)
                Val = []
                for layer in LayerNames:
                    Total_score[layer] = prev + lambda1 * TAU1[layer] +  lambda2 * TAU2[layer] + lambda3 * TAU3[layer]
                    prev= Total_score[layer]
                Val = [Total_score[layer]]
                writer.writerow(Val)
            final_Scores += [prev]
            TAU3List[count] = TAU3
            TAU2List[count] = TAU2
            TAU3List[count] = TAU1
            if count %100 == 0:
                print(count)
            if count == 3000:
                break
    # with open('/home/local/ASUAD/asing651/ResnetCifar10pytorchFI/DRDNA/b10c32/TAU1nosdc.pkl', 'wb') as f:
    #     pickle.dump(TAU1 ,f)
    # with open('/home/local/ASUAD/asing651/ResnetCifar10pytorchFI/DRDNA/b10c32/TAU2nosdc.pkl', 'wb') as f:
    #     pickle.dump(TAU2 ,f)
    # with open('/home/local/ASUAD/asing651/ResnetCifar10pytorchFI/DRDNA/b10c32/TAU3nosdc.pkl', 'wb') as f:
    #     pickle.dump(TAU3 ,f)
    print(f'\nTest set: Average loss: {test_loss/len(testloader):.4f}, Accuracy: {correct}/{total} ({100.*correct/total:.2f}%) , Wrong = {wrong}\n')

if __name__ == '__main__':

    net = resnet18(pretrained=True, progress=True)
    num_ftrs = net.fc.in_features
    #net.fc = nn.Linear(num_ftrs, 10)  # Modify the last layer to output 10 classes
    net = net.to(device)

    '''
    commented below code for testing without fault injection
    '''
    # batch_size = 1
    # H = 32
    # W = 32
    # C = 3
    # bit_pos = 1
    # ranges = [9999,9999,9999,9999,9999,9999,999999999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999]
    # pfi = single_bit_flip_func(
    #             net,
    #             batch_size=batch_size,
    #             input_shape=[C,H,W],
    #             use_cuda=True,
    #             bits=8,
    #             random_batch=False,
    #             bit_pos=bit_pos,
    #         )

    # fi_layer=5
    # fi_c = 6
    # fi_h = 7
    # fi_w = 2

    # corrupt_model = random_neuron_single_bit_inj_Aman(pfi, ranges, fi_layer, fi_c, fi_h, fi_w,bit_pos = bit_pos)
    selected_neurons = []
    data = torch.load('correct_data.pt')
    correct_inputs = data['inputs']
    correct_targets = data['targets']
    new_test_dataset = TensorDataset(correct_inputs, correct_targets)

    # Create a DataLoader
    testloader = DataLoader(
        new_test_dataset,
        batch_size=1,  # Use the batch size you need
        shuffle=False,
        num_workers=0  # Adjust based on your environment
    )

    with open(pathtoDetectionSites, 'r') as f:
        for line in f:
            layer_name, location = line.strip().split(',',1)
            location = eval(location.strip()) 
            selected_neurons.append((layer_name, location))
            if layer_name not in LayerNames:
                LayerNames += [layer_name]

        with open(pathtoTau1, 'rb') as f:
            tau1 = pickle.load(f)
        with open(pathtoTau2, 'rb') as f:
            tau2 = pickle.load(f)
        with open(pathtoTau3, 'rb') as f:
            tau3  =pickle.load(f)
        T2 = copy.deepcopy(tau2)
        T1 = copy.deepcopy(tau1)
        T3 = copy.deepcopy(tau3)
        for layer_name in T2:
            reset_histogram(T2[layer_name])

        # test_without_fault(corrupt_model,T1,T2,T3)
        test_without_fault(net,T1,T2,T3)
