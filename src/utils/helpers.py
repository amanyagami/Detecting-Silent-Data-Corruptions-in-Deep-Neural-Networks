import torch

"""Compare two tensors element-wise and return positions of differences."""


def compare_tensors(tensor1, tensor2):

    if tensor1.size() != tensor2.size():
        return f"Different sizes: {tensor1.size()} vs {tensor2.size()}"

    differences = (tensor1 != tensor2).nonzero(as_tuple=False)
    if differences.size(0) == 0:
        return None  # No differences

    diff_list = differences.tolist()
    return diff_list


""" 
find_differences takes two dictionarys of tensors and find the location of difference where
the two tensors dffer in value
key : model layer 
value = tensor location of 
example output - Differences: {'avgpool': [[0, 0, 0, 0]], 'fc': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9]]}
"""


def find_differences(d1, d2):
    differences = {}

    # Check if both dictionaries have the same keys
    if d1.keys() != d2.keys():
        keys_in_d1_not_in_d2 = d1.keys() - d2.keys()
        keys_in_d2_not_in_d1 = d2.keys() - d1.keys()
        if keys_in_d1_not_in_d2:
            differences["keys_in_d1_not_in_d2"] = list(keys_in_d1_not_in_d2)
        if keys_in_d2_not_in_d1:
            differences["keys_in_d2_not_in_d1"] = list(keys_in_d2_not_in_d1)

    # Compare tensors for common keys
    for key in d1.keys() & d2.keys():
        diff_positions = compare_tensors(d1[key], d2[key])
        if diff_positions is not None:
            differences[key] = diff_positions

    return differences


"""
get_activation function is used to attach hook in the model
"""
activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()

    return hook


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
