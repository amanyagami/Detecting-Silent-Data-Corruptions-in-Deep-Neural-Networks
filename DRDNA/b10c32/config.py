from scipy.stats import wasserstein_distance
from bisect import bisect_left
import torch

bins_num = 10
cohortSize = 32
Lambda1 = 1
Lambda2 = 1
Lambda3 = 1

import os

# NOTE: was a hardcoded absolute path to the original author's machine;
# resolved relative to this file so the package works on any machine.
path = os.path.dirname(os.path.abspath(__file__))
def update_histogram(value, histogram):
    sorted_keys = list(histogram.keys())
    first_bin_range = sorted_keys[0]
    last_bin_range = sorted_keys[-1]

    # Handle values less than the first bin
    if value < first_bin_range[0]:
        histogram[first_bin_range] += 1
        return histogram
    # Handle values greater than or equal to the last bin
    elif value >= last_bin_range[1]:
        histogram[last_bin_range] += 1
        return histogram
    # Handle values within the bin ranges
    else:
        for bin_range in sorted_keys:
            if bin_range[0] <= value < bin_range[1]:
                histogram[bin_range] += 1
                return histogram



def reset_histogram(histogram):
    # Set all frequencies to zero, keeping the bin keys
    return {key: 0 for key in histogram}

def normalize_histogram(histogram):
    # Calculate the total count of all bins
    total_count = sum(histogram.values())
    
    # Normalize each bin frequency
    if total_count > 0:  # To avoid division by zero
        return {key: count / total_count for key, count in histogram.items()}
    else:
        return histogram  # If the total count is zero, return the histogram unchanged


def listtohistogram(data,bins= 10):
        # Calculate bin edges for 10 bins
    min_data = min(data)
    max_data = max(data)
    range_data = max_data - min_data
    bin_width = range_data / bins
    bin_edges = [min_data + i * bin_width for i in range(bins+1)]

    # Function to determine bin index
    def find_bin(value, edges):
        return bisect_left(edges, value) - 1

    # Count frequencies per bin
    bin_count = [0] * bins
    for value in data:
        bin_index = find_bin(value, bin_edges)
        bin_count[bin_index] += 1

    # Store in dictionary with bin ranges as keys
    histogram_dict = {
        (bin_edges[i] , bin_edges[i+1]): bin_count[i]
        for i in range(len(bin_edges) - 1)
    }
    return histogram_dict

def TAU3processing(tau3, count):
    tau3_activation_extremes = {}
    for layer_name in tau3:
        tau3[layer_name] /= count
        tensor= tau3[layer_name]
        flat_tensor = tensor.flatten()
        max_value, max_index = flat_tensor.max(0)
        min_value, min_index = flat_tensor.min(0)
        
        # Convert indices to tensor before using unravel_index
        max_index_tensor = torch.tensor(max_index.item())
        min_index_tensor = torch.tensor(min_index.item())
        
        # Get the index in the original tensor dimensions
        max_location = torch.unravel_index(max_index_tensor, tensor.shape)
        min_location = torch.unravel_index(min_index_tensor, tensor.shape)
        
        # Store the results in the output dictionary
        tau3_activation_extremes[layer_name] = {
            'max_value': max_value.item(),
            'max_location': max_location,
            'min_value': min_value.item(),
            'min_location': min_location
        }
    return tau3_activation_extremes
def TAU2processing(tau2):
    tau2histtodict = {}
    for layer_name in tau2:
        # print(len(tau2[layer_name])," num of neuron activation in tau1")
        tau2histtodict[layer_name] = listtohistogram(tau2[layer_name])
    return tau2histtodict
def  abnormility_score1(x,histogram_data):
      # Iterate through the bins (key ranges) and their frequencies
    for (min_val, max_val), frequency in histogram_data.items():
        if min_val <= x < max_val:  # Check if x falls within this bin's range
            return (1-frequency)
    # If no bin is found, abnormality score 1
    return 1

def abnormality_score2(histo1 , histo2):
    bins = [(a + b) / 2 for a, b in histo1.keys()]
    weights1 = list(histo1.values())
    weights2 = list(histo2.values())
    total = sum(weights1) * sum(weights2)
    # Calculate Earth Mover's Distance
    emd_value = wasserstein_distance(bins, bins, weights1, weights2)
    # normalize

    return emd_value/total


def abnormality_score3(a,b):
    if a['max_location'] == b['max_location'] or a['min_location'] == b['max_location']:
        if a['max_location'] == b['max_location'] and a['min_location'] == b['max_location']:
            return 1
        return 0.5
    return 0

