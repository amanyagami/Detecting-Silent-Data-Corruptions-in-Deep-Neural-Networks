
import torch
from src.utils.customFI_methods import single_bit_flip_func
from models.resnet import resnet18
from utils.helpers import get_activation
from utils.customFI_methods import random_neuron_single_bit_inj_ours
# Load pre-trained ResNet-18 model
net = resnet18(pretrained=True, progress=True)
num_ftrs = net.fc.in_features

batch_size = 1
H = 32
W = 32
C = 3
ranges = [9999,9999,9999,9999,9999,9999,999999999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999]#,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999]
pfi = single_bit_flip_func(
            net,
            batch_size=batch_size,
            input_shape=[C,H,W],
            use_cuda=torch.cuda.is_available(),  # NOTE: was hardcoded True; crashed on any CPU-only machine
            bits=8,
            random_batch=False
        )

fi_layer=19
fi_c = 0
fi_h = 0
fi_w = 0
corrupt_model = random_neuron_single_bit_inj_ours(pfi, ranges, fi_layer, fi_c, fi_h, fi_w)
corrupt_model.conv1.register_forward_hook(get_activation('conv1'))
corrupt_model.bn1.register_forward_hook(get_activation('bn1'))
corrupt_model.layer1[0].conv1.register_forward_hook(get_activation('layer1.0.conv1'))
corrupt_model.layer1[0].bn1.register_forward_hook(get_activation('layer1.0.bn1'))
corrupt_model.layer1[0].conv2.register_forward_hook(get_activation('layer1.0.conv2'))
corrupt_model.layer1[0].bn2.register_forward_hook(get_activation('layer1.0.bn2'))
corrupt_model.layer2[0].conv1.register_forward_hook(get_activation('layer2.0.conv1'))
corrupt_model.layer2[0].bn1.register_forward_hook(get_activation('layer2.0.bn1'))
corrupt_model.layer2[0].conv2.register_forward_hook(get_activation('layer2.0.conv2'))
corrupt_model.layer2[0].bn2.register_forward_hook(get_activation('layer2.0.bn2'))
corrupt_model.layer3[0].conv1.register_forward_hook(get_activation('layer3.0.conv1'))
corrupt_model.layer3[0].bn1.register_forward_hook(get_activation('layer3.0.bn1'))
corrupt_model.layer3[0].conv2.register_forward_hook(get_activation('layer3.0.conv2'))
corrupt_model.layer3[0].bn2.register_forward_hook(get_activation('layer3.0.bn2'))
corrupt_model.layer4[0].conv1.register_forward_hook(get_activation('layer4.0.conv1'))
corrupt_model.layer4[0].bn1.register_forward_hook(get_activation('layer4.0.bn1'))
corrupt_model.layer4[0].conv2.register_forward_hook(get_activation('layer4.0.conv2'))
corrupt_model.layer4[0].bn2.register_forward_hook(get_activation('layer4.0.bn2'))
corrupt_model.avgpool.register_forward_hook(get_activation('avgpool'))
corrupt_model.fc.register_forward_hook(get_activation('fc'))

corrupt_model.eval()