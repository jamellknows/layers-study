import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
from collections import Counter
import os
from PIL import Image
import torch.nn.functional as F
import torch.nn.init as init
import math




# Custom Initialisations
#Xavier Uniform
def init_xavier_uniform(tensor):
    return nn.init.xavier_uniform(tensor)

# He Initialization
def init_kaiming_uniform(tensor):
    return nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity='relu')

# Zero Initialization
def init_zeros(tensor):
    return nn.init.zeros_(tensor)

# Normal Initialization
def init_normal(tensor):
    return nn.init.normal_(tensor, mean=0, std=1)

# Orthogonal Initialization
def init_orthogonal(tensor):
    return nn.init.orthogonal_(tensor, gain=1)

# Uniform Initialization
def init_uniform(tensor):
    return nn.init.uniform_(tensor, a=0, b=1)

# Sparse Initialization
def init_sparse(tensor):
    return nn.init.sparse_(tensor, sparsity=0.1, std=0.01)

# Required Miscellaneous Functions

def replicate_max_value(input_tensor):
    # Find the maximum value in the input tensor
    max_value = torch.max(input_tensor).item()

    # Create a new tensor with the same length as input_tensor, filled with the maximum value
    replicated_tensor = torch.full_like(input_tensor, max_value)

    return replicated_tensor

def is_square_matrix(matrix):
    if not torch.is_tensor(matrix):
        raise ValueError("Input is not a PyTorch tensor.")

    # Check if the tensor has two dimensions
    if len(matrix.shape) != 2:
        return False

    # Check if the number of rows is equal to the number of columns
    return matrix.shape[0] == matrix.shape[1]


def split_and_pad_tensor(input_tensor, num_chunks, dim=0):
    # Calculate the size of each chunk
    try:
        chunk_size = input_tensor.shape[dim] // num_chunks
    except IndexError:
    # Handle the IndexError by setting chunk_size to 1
        chunk_size = 1

    # Calculate the size of the last chunk
    try:
        last_chunk_size = input_tensor.shape[dim] % num_chunks
    except IndexError:
    # Handle the IndexError by setting last_chunk_size to 1
        last_chunk_size = 1

    # Split the tensor into chunks
    try:
    # Your code that might raise RuntimeError
    # ...
        split_tensors = torch.chunk(input_tensor, num_chunks, dim=dim)

    except RuntimeError as e:
        print(f"RuntimeError occurred with input_tensor: {input_tensor}")
    # Handle the error as needed
    # ...
    split_tensors = torch.chunk(input_tensor, num_chunks, dim=dim)

    # Pad the last chunk with zeros if needed
    if last_chunk_size > 0:
        padding_size = chunk_size - last_chunk_size
        padding = torch.zeros_like(split_tensors[-1][:padding_size], device=input_tensor.device)
        split_tensors = list(split_tensors)
        split_tensors[-1] = torch.cat([split_tensors[-1], padding], dim=dim)
        split_tensors = tuple(split_tensors)


    return split_tensors


def calculate_num_chunks(tensor_size, split_dim=0):
    # Check if the input tensor_size is a torch.Size object or a tuple
    if isinstance(tensor_size, torch.Size):
        # Get the size along the specified dimension
        try:
            dim_size = tensor_size[split_dim]
        except IndexError:
        # Handle the IndexError by setting sim_size to 1
            dim_size = 1
        else:
    # If no IndexError occurred, calculate sim_size as usual
            dim_size = (dim_size + 2) // 3  # Adding 2 to handle remainder
    elif isinstance(tensor_size, tuple) and all(torch.is_tensor(t) for t in tensor_size):
        # If it's a tuple of tensors, get the size along the specified dimension for each tensor
        dim_size = tensor_size[0].size(split_dim)

    else:
        raise ValueError("Input tensor_size should be a torch.Size object or a tuple of tensors.")

    # Calculate the number of chunks
    num_chunks = (dim_size + 2) // 3  # Adding 2 to handle remainder

    return num_chunks
def pad_or_repeat_to_match_multiply(set1, set2):
    # Determine the length of the sets
    max_length = max(len(set1), len(set2))

    # Extend the shorter set by repeating its elements
    set1 = set1 * (max_length // len(set1)) + set1[:max_length % len(set1)]
    set2 = set2 * (max_length // len(set2)) + set2[:max_length % len(set2)]

    # Pad or repeat tensors in set2 along dimension 0 to match the size of set1
    for i in range(len(set1)):
        if set1[i].size(0) > set2[i].size(0):
            padding_size = set1[i].size(0) - set2[i].size(0)
            if padding_size % 2 == 1:  # If padding size is odd, pad with zeros
                set2[i] = torch.cat([set2[i], torch.zeros(1, *set2[i].shape[1:])], dim=0) if set2[i].size(0) % 2 == 1 else torch.cat([torch.zeros(1, *set2[i].shape[1:]), set2[i], torch.zeros(1, *set2[i].shape[1:])], dim=0)
            else:  # If padding size is even, repeat the tensor
                set2[i] = set2[i].repeat(padding_size // 2 + 1, 1, 1)

    # Multiply corresponding tensors iteratively
    result = [tensor1 * tensor2 for tensor1, tensor2 in zip(set1, set2)]

    return result

def multiply_iteratively(set1, set2):
    # Determine the length of the longer set
    max_length = max(len(set1), len(set2))

    # Extend the shorter set by repeating its elements
    set1 = set1 * (max_length // len(set1)) + set1[:max_length % len(set1)]
    set2 = set2 * (max_length // len(set2)) + set2[:max_length % len(set2)]

    # Pad the second set along dimension 0 to match the size of the first set
    if set1[0].size(0) > set2[0].size(0):
        padding_size = set1[0].size(0) - set2[0].size(0)
        set2 = [torch.cat([tensor, torch.zeros(padding_size, *tensor.shape[1:])], dim=0) for tensor in set2]
    for i in range(0, len(set1)):
        print(f'set 1 size {set1[i].size()}')
        print(f'set 2 size {set2[i].size()}')
    
    # Multiply corresponding tensors iteratively
    result = []
    for i in range(0, len(set1)):
        result.append(set1[i] * set2[i])

    return torch.stack(result)

def calculate_angle(tensor1, tensor2):
    # Flatten tensors to vectors

    vector1 = tensor1
    vector2 = tensor2
    vector1 = vector1.to(dtype=torch.float32)
    vector2 = vector2.to(dtype=torch.float32)
    # Calculate dot product
  
  
        
    dot_product = matmul_or_transpose(vector1, vector2)

    # Calculate magnitudes
    

    magnitude1 = torch.norm(vector1)
    magnitude2 = torch.norm(vector2)

    # Calculate cosine of the angle
    cosine_angle = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians
    angle_rad = torch.acos(torch.clamp(cosine_angle, -1.0, 1.0))
    
    angle_rad = torch.tensor(angle_rad, requires_grad=True)

    # Convert angle to degrees
    angle_deg = torch.rad2deg(angle_rad)

    return angle_rad 

# #########################################################################
# Padding Functions


def reshape_tensor(input_tensor, target_shape):
    """
    Reshape a tensor to the given target shape, removing or padding elements as needed.

    Parameters:
    - input_tensor: torch.Tensor, the input tensor to be reshaped.
    - target_shape: tuple, the target shape for the tensor.

    Returns:
    - reshaped_tensor: torch.Tensor, the reshaped tensor.
    """

    # Flatten the input tensor and calculate the total number of elements
    flat_tensor = input_tensor.view(-1)
    total_elements = flat_tensor.numel()

    # Calculate the total number of elements in the target shape
    target_elements = torch.prod(torch.tensor(target_shape))

    # If the total elements match, reshape directly
    if total_elements == target_elements:
        reshaped_tensor = flat_tensor.view(*target_shape)
    elif total_elements < target_elements:
        # If fewer elements, pad with zeros
        pad_elements = target_elements - total_elements
        pad_values = torch.zeros(pad_elements, dtype=flat_tensor.dtype, device=flat_tensor.device)
        reshaped_tensor = torch.cat((flat_tensor, pad_values), dim=0).view(*target_shape)
    else:
        # If more elements, truncate
        reshaped_tensor = flat_tensor[:target_elements].view(*target_shape)

    return reshaped_tensor

def reshape_tensor_to_match_one(target_tensor, reference_tensor):
    """
    Reshape the target tensor to match the shape of the reference tensor.

    Parameters:
    - target_tensor: torch.Tensor, the tensor to be reshaped.
    - reference_tensor: torch.Tensor, the tensor whose shape should be matched.

    Returns:
    - reshaped_tensor: torch.Tensor, the reshaped tensor.
    """

    # Get the size of the target tensor
    target_size = target_tensor.numel()

    # Get the total number of elements in the reference tensor
    reference_size = reference_tensor.numel()

    if target_size < reference_size:
        # Repeat the values to match the size of the reference tensor
        repeated_tensor = target_tensor.repeat(reference_size // target_size + 1)
        # Slice to match the size exactly
        reshaped_tensor = repeated_tensor[:reference_size].view_as(reference_tensor)
    elif target_size > reference_size:
        # Remove excess values to match the size of the reference tensor
        reshaped_tensor = target_tensor[:reference_size].view_as(reference_tensor)
    else:
        # Sizes are already equal, no need to reshape
        reshaped_tensor = target_tensor.view_as(reference_tensor)

    return reshaped_tensor

def reshape_tensor_to_match_two(target_tensor, reference_tensor, dim=0):
    """
    Reshape the target tensor to match the shape of the reference tensor along the specified dimension.

    Parameters:
    - target_tensor: torch.Tensor, the tensor to be reshaped.
    - reference_tensor: torch.Tensor, the tensor whose shape should be matched.
    - dim: int, the dimension along which to reshape.

    Returns:
    - reshaped_tensor: torch.Tensor, the reshaped tensor.
    """

    # Get the size of the target tensor along the specified dimension
    target_size = target_tensor.size(dim)

    # Get the size of the reference tensor along the specified dimension
    reference_size = reference_tensor.size(dim)

    if target_size < reference_size:
        # Repeat the values along the specified dimension to match the size of the reference tensor
        repeated_tensor = target_tensor.repeat(*(1 if i != dim else reference_size // target_size for i in range(target_tensor.dim())))
        # Slice to match the size exactly
        reshaped_tensor = repeated_tensor.index_select(dim, torch.arange(reference_size))
    elif target_size > reference_size:
        # Remove excess values along the specified dimension to match the size of the reference tensor
        reshaped_tensor = target_tensor.index_select(dim, torch.arange(reference_size))
    else:
        # Sizes are already equal along the specified dimension, no need to reshape
        reshaped_tensor = target_tensor

    return reshaped_tensor

def reshape_tensor_to_match_three(target_tensor, reference_tensor):
    """
    Reshape the target tensor to match the shape of the reference tensor.

    Parameters:
    - target_tensor: torch.Tensor, the tensor to be reshaped.
    - reference_tensor: torch.Tensor, the tensor whose shape should be matched.

    Returns:
    - reshaped_tensor: torch.Tensor, the reshaped tensor.
    """

    # Get the size of the target tensor
    target_size = target_tensor.numel()

    # Get the total number of elements in the reference tensor
    reference_size = reference_tensor.numel()

    if target_size < reference_size:
        # Repeat the values to match the size of the reference tensor
        repeated_tensor = target_tensor.unsqueeze(0).repeat(reference_size // target_size + 1, 1)
        # Slice to match the size exactly
        reshaped_tensor = repeated_tensor[:, :reference_size].view_as(reference_tensor)
    elif target_size > reference_size:
        # Remove excess values to match the size of the reference tensor
        reshaped_tensor = target_tensor[:reference_size].view_as(reference_tensor)
    else:
        # Sizes are already equal, no need to reshape
        reshaped_tensor = target_tensor.view_as(reference_tensor)

    return reshaped_tensor

def reshape_tensor_to_smaller_size(target_tensor, reference_tensor):
    """
    Reshape the target tensor to a smaller size defined by the reference tensor.

    Parameters:
    - target_tensor: torch.Tensor, the tensor to be reshaped.
    - reference_tensor: torch.Tensor, the tensor whose size should be used as the reference.

    Returns:
    - reshaped_tensor: torch.Tensor, the reshaped tensor.
    """
    
    
    # Get the size of the reference tensor
    reference_size = reference_tensor.numel()

    # Ensure that the target size is greater than or equal to the reference size
    if target_tensor.numel() < reference_size:
        raise ValueError("Target tensor size should be greater than or equal to the reference size.")

    # Use boolean indexing to keep only the elements within the reference size
    reshaped_tensor = target_tensor[:reference_size]

    return reshaped_tensor

def align_tensors(tensor1, tensor2):
    """
    Align two tensors by slicing the larger tensor to match the size of the smaller one along each axis.

    Parameters:
    - tensor1 (torch.Tensor): First input tensor.
    - tensor2 (torch.Tensor): Second input tensor.

    Returns:
    - aligned_tensor1 (torch.Tensor): Aligned tensor1.
    - aligned_tensor2 (torch.Tensor): Aligned tensor2.
    """
    # Get the shapes of the input tensors
    shape1 = tensor1.shape
    shape2 = tensor2.shape

    # Determine the minimum size along each axis
    target_size = tuple(min(size1, size2) for size1, size2 in zip(shape1, shape2))

    # Slice the tensors to the target size along each axis
    aligned_tensor1 = tensor1[tuple(slice(0, size) for size in target_size)]
    aligned_tensor2 = tensor2[tuple(slice(0, size) for size in target_size)]

    return aligned_tensor1, aligned_tensor2

def truncate_to_square(tensor):
    """
    Truncate a tensor to a square by removing elements at the end of the longer axis.

    Parameters:
    - tensor (torch.Tensor): Input tensor.

    Returns:
    - square_tensor (torch.Tensor): Truncated square tensor.
    """
    # Determine the size along each axis
    size_0, size_1 = tensor.size()

    # Find the minimum size to truncate to a square
    min_size = min(size_0, size_1)

    # Truncate the tensor to a square along both axes
    square_tensor = tensor[:min_size, :min_size]

    return square_tensor


def repeat_to_match(tensor1, tensor2):
    """
    Repeat the second tensor along both axes to match the smallest length of the first tensor.

    Parameters:
    - tensor1 (torch.Tensor): First input tensor.
    - tensor2 (torch.Tensor): Second input tensor.

    Returns:
    - repeated_tensor2 (torch.Tensor): Second tensor repeated along both axes to match the smallest length of the first tensor.
    """
    # Find the minimum size along both axes
    min_size_0 = min(tensor1.size(0), tensor2.size(0))
    min_size_1 = min(tensor1.size(1), tensor2.size(1))

    # Determine the number of repetitions needed along each axis
    repeat_0 = tensor1.size(0) // min_size_0
    repeat_1 = tensor1.size(1) // min_size_1

    # Repeat the second tensor along both axes to match the smallest length of the first tensor
    repeated_tensor2 = tensor2[:min_size_0, :min_size_1].repeat(repeat_0, repeat_1)

    return repeated_tensor2

def repeat_elements_to_match_shape(input_tensor, target_shape):
    """
    Repeat the elements of the input tensor until its shape matches the target shape.

    Parameters:
    - input_tensor: torch.Tensor, the input tensor to be repeated.
    - target_shape: tuple, the target shape to match.

    Returns:
    - repeated_tensor: torch.Tensor, the tensor with repeated elements.
    """

    # Get the size of the target tensor
    target_size = torch.tensor(target_shape).prod().item()
    # Repeat the values to match the size of the target tensor
    repeated_tensor = input_tensor.repeat(target_size // input_tensor.numel() + 1)

    # Slice to match the size exactly
    repeated_tensor = repeated_tensor[:target_size]

    # Reshape the tensor to the target shape
    repeated_tensor = repeated_tensor.view(*target_shape)

    return repeated_tensor

def repeat_elements_to_match_shape_2d(input_tensor, target_shape):
    """
    Repeat the elements of the 2D input tensor until its shape matches the target shape.

    Parameters:
    - input_tensor: torch.Tensor, the 2D input tensor to be repeated.
    - target_shape: tuple, the target shape to match (2D).

    Returns:
    - repeated_tensor: torch.Tensor, the 2D tensor with repeated elements.
    """

    if input_tensor.dim() != 2 or len(target_shape) != 2:
        raise ValueError("Input and target shape must be 2D.")

    # Get the dimensions of the input tensor
    input_rows, input_cols = input_tensor.size()

    # Get the dimensions of the target tensor
    target_rows, target_cols = target_shape

    # Repeat along rows and columns
    repeated_rows = input_tensor.repeat(target_rows // input_rows, 1)
    repeated_cols = repeated_rows.repeat(1, target_cols // input_cols)

    # Slice to match the size exactly
    repeated_tensor = repeated_cols[:target_rows, :target_cols]

    return repeated_tensor

def pad_smaller_tensor(tensor1, tensor2):
    """
    Pad the smaller tensor to match the larger tensor's sizes on all axes and return the new padded tensor.

    Parameters:
    - tensor1 (torch.Tensor): First input tensor.
    - tensor2 (torch.Tensor): Second input tensor.

    Returns:
    - padded_tensor (torch.Tensor): Smaller tensor padded to match the sizes of the larger tensor.
    """
    # Determine the sizes of both tensors
    size1 = tensor1.size()
    size2 = tensor2.size()

    # Find the maximum size along each axis
    max_size = [max(size1[i], size2[i]) for i in range(max(len(size1), len(size2)))]

    # Calculate the padding needed for each tensor
    padding1 = [max_size[i] - size1[i] for i in range(len(max_size))]
    padding2 = [max_size[i] - size2[i] for i in range(len(max_size))]

    # Determine which tensor is smaller
    smaller_tensor = tensor1 if torch.numel(tensor1) < torch.numel(tensor2) else tensor2

    # Apply zero-padding to the smaller tensor
    padded_tensor = torch.nn.functional.pad(smaller_tensor, pad=(0,) * (len(max_size) * 2))

    return padded_tensor


def square_tensor_by_padding(input_tensor):
    # Get the dimensions of the input tensor
    original_size = input_tensor.size()
    
    # Find the maximum dimension
    max_dimension = max(original_size)
    
    # Calculate the padding needed for each dimension
    padding = [max_dimension - size for size in original_size]
    
    # Apply zero-padding to the input tensor
    padded_tensor = torch.nn.functional.pad(input_tensor, (0, padding[1], 0, padding[0]))
    
    return padded_tensor

def pad_to_shape(input_tensor, target_shape):
    """
    Pad the input tensor to the specified target shape.

    Parameters:
    - input_tensor: torch.Tensor, the input tensor to be padded.
    - target_shape: tuple, the target shape to which the tensor should be padded.

    Returns:
    - padded_tensor: torch.Tensor, the padded tensor.
    """

    # Calculate the padding for each dimension
    padding = [max(0, dim_size - input_size) for input_size, dim_size in zip(input_tensor.shape, target_shape)]

    # Pad the tensor
    padded_tensor = F.pad(input_tensor, padding)

    return padded_tensor



def pad_tensor_to_shape(tensor, target_shape):
    """
    Pad a tensor to match the given target shape by expanding or adding axes as needed.

    Parameters:
    - tensor (torch.Tensor): Input tensor.
    - target_shape (tuple): Target shape to pad the tensor to.

    Returns:
    - padded_tensor (torch.Tensor): Tensor padded to match the target shape.
    """
    # Get the current size of the input tensor
    current_size = tensor.size()

    # Determine the dimensionality of the tensor
    tensor_dim = len(current_size)

    # Determine the number of axes to add or expand
    num_axes_to_add = max(len(target_shape) - tensor_dim, 0)

    # Pad the tensor by expanding or adding axes as needed
    padded_tensor = tensor.unsqueeze(-1).expand(*current_size, *([1] * num_axes_to_add))

    return padded_tensor


def pad_1d_tensor(input_tensor, target_length, pad_value=0):
    """
    Pad a 1D tensor to a specified length.

    Parameters:
    - input_tensor: torch.Tensor, the input 1D tensor to be padded.
    - target_length: int, the desired length of the padded tensor.
    - pad_value: int or float, the value used for padding.

    Returns:
    - padded_tensor: torch.Tensor, the padded 1D tensor.
    """

    current_length = input_tensor.size(0)

    if current_length == target_length:
        return input_tensor.clone()

    # Calculate the amount of padding needed on both sides
    pad_before = (target_length - current_length) // 2
    pad_after = target_length - current_length - pad_before

    # Use torch.nn.functional.pad to pad the tensor
    padded_tensor = torch.nn.functional.pad(input_tensor, (pad_before, pad_after), value=pad_value)

    return padded_tensor

def copy_elements_to_target_shape_1d(input_tensor, target_shape):
    """
    Copy the elements of the 1D input tensor to create a new tensor with the target shape.

    Parameters:
    - input_tensor: torch.Tensor, the 1D input tensor to be copied.
    - target_shape: int, the target length of the resulting tensor.

    Returns:
    - copied_tensor: torch.Tensor, the 1D tensor with copied elements.
    """

    if input_tensor.dim() != 1 or len(target_shape) != 1:
        raise ValueError("Input and target shape must be 1D.")

    input_length = input_tensor.size(0)
    target_length = target_shape[0]

    # Repeat along the length
    repeated_tensor = input_tensor.repeat(target_length // input_length + 1)

    # Slice to match the size exactly
    copied_tensor = repeated_tensor[:target_length]

    return copied_tensor

def copy_elements_to_target_shape_2D(input_tensor, target_shape):
    """
    Copy the elements of the 2D input tensor to create a new tensor with the target shape.

    Parameters:
    - input_tensor: torch.Tensor, the 2D input tensor to be copied.
    - target_shape: tuple, the target shape to match (2D).

    Returns:
    - copied_tensor: torch.Tensor, the 2D tensor with copied elements.
    """

    if input_tensor.dim() != 2 or len(target_shape) != 2:
        raise ValueError("Input and target shape must be 2D.")

    input_rows, input_cols = input_tensor.size()
    target_rows, target_cols = target_shape

    # Repeat along rows and columns
    repeated_rows = input_tensor.repeat(target_rows // input_rows, 1)
    repeated_cols = repeated_rows.repeat(1, target_cols // input_cols)

    # Slice to match the size exactly
    copied_tensor = repeated_cols[:target_rows, :target_cols]

    return copied_tensor

def copy_elements_to_target_shape(input_tensor, target_shape):
    """
    Copy the elements of the 1D input tensor to create a new tensor with the target shape.

    Parameters:
    - input_tensor: torch.Tensor, the 1D input tensor to be copied.
    - target_shape: torch.Size, the target shape of the resulting tensor.

    Returns:
    - copied_tensor: torch.Tensor, the tensor with copied elements to match the target shape.
    """

    if input_tensor.dim() != 1 or len(target_shape) != 2:
        raise ValueError("Input tensor must be 1D and target shape must be 2D.")

    input_length = input_tensor.size(0)
    target_rows, target_cols = target_shape

    # Repeat along the axis with smaller size
    repeated_tensor = input_tensor.repeat(target_rows // input_length + 1)

    # Slice to match the size exactly
    copied_tensor = repeated_tensor[:target_rows].view(target_shape)

    return copied_tensor

def repeat_and_pad_to_shape(input_tensor, target_shape):
    """
    Repeat and pad a 2D tensor to match a target shape.

    Parameters:
    - input_tensor: torch.Tensor, the input 2D tensor.
    - target_shape: tuple, the target shape to match.

    Returns:
    - torch.Tensor, the repeated and padded tensor.
    """
    if input_tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D.")

    input_rows, input_cols = input_tensor.size()
    target_rows, target_cols = target_shape

    # Repeat rows and cols to match target shape
    repeated_rows = input_tensor.repeat(target_rows // input_rows, 1)
    repeated_cols = repeated_rows[:, :target_cols]

    # Pad if needed
    if target_cols > input_cols:
        pad_cols = target_cols - input_cols
        padding = torch.ones(repeated_cols.size(0), pad_cols) * torch.sqrt(torch.tensor(2.0))
        repeated_cols = torch.cat([repeated_cols, padding], dim=1)

    return repeated_cols

def matmul_or_transpose(tensor1, tensor2):

    """
    Multiply two tensors using matmul or transpose the second tensor if needed.

    Parameters:
    - tensor1: torch.Tensor, the first tensor.
    - tensor2: torch.Tensor, the second tensor.

    Returns:
    - torch.Tensor, the result of the multiplication.
    """
    if tensor1.dim() == 0:
        
        return torch.matmul(tensor2, tensor2.t())
    if tensor2.dim() == 0:
        
        return torch.matmul(tensor1, tensor1.t())
    if len(tensor1.size()) == 1:
        return torch.matmul(tensor1, tensor2)
    result = 0
    tensor1 = tensor1.to(dtype=torch.float32)
    tensor2 = tensor2.to(dtype=torch.float32)
    nope = True
    if len(tensor1.shape) == 4:
        tensor2 = tensor2.squeeze(0).squeeze(0)
        result = torch.matmul(tensor1, tensor2.t())
  
      
    
    # elif tensor1.dim() != 2 and nope == True or tensor2.dim() != 2 and nope == True:
    #     raise ValueError("Both tensors must be 2D.")

    # Check if the tensors can be multiplied directly
    if tensor1.size(1) == tensor2.size(0):
        result = torch.matmul(tensor1, tensor2)
    elif tensor1.size(1) == tensor2.size(1):
        # Transpose the second tensor
        result = torch.matmul(tensor1, tensor2.t())
    return result


def subtract_tensors(t1, t2):
    """
    Attempt to subtract t2 from t1, handling size mismatches, transposing, padding, and repeating.

    Parameters:
    - t1: torch.Tensor, the first tensor.
    - t2: torch.Tensor, the second tensor.

    Returns:
    - torch.Tensor, the result of the subtraction.
    """
    if t1.size() == t2.size():
        result = t1 - t2
    elif t1.size() == torch.Size([t2.size(0), t2.size(1)]):
        result = t1 - t2
    elif t1.size() == torch.Size([t2.size(1), t2.size(0)]):
        result = t1 - t2.t()
    else:
        # Try padding and subtracting
        padded_t2 = subtract_pad_tensor(t2, t1.size())
        result = t1 - padded_t2

        if result.size() != t1.size():
            # Try repeating and subtracting
            repeated_t2 = subtract_repeat_elements_to_match_shape(t2, t1.size())
            result = t1 - repeated_t2

    return result

def subtract_pad_tensor(input_tensor, target_shape):
    """
    Pad the input tensor to match the target shape.

    Parameters:
    - input_tensor: torch.Tensor, the input tensor.
    - target_shape: torch.Size, the target shape.

    Returns:
    - torch.Tensor, the padded tensor.
    """
    padded_tensor = torch.nn.functional.pad(input_tensor, (0, target_shape[1] - input_tensor.size(1)))
    return padded_tensor

def subtract_repeat_elements_to_match_shape(input_tensor, target_shape):
    """
    Repeat the elements of the input tensor to match the target shape.

    Parameters:
    - input_tensor: torch.Tensor, the input tensor.
    - target_shape: torch.Size, the target shape.

    Returns:
    - torch.Tensor, the repeated tensor.
    """
    repeated_tensor = input_tensor.repeat(target_shape[0] // input_tensor.size(0) + 1, target_shape[1] // input_tensor.size(1) + 1)
    return repeated_tensor[:target_shape[0], :target_shape[1]]

def ensure_size(obj):
    """
    Ensure that the input is a torch.Size object. If not, convert it to torch.Size.

    Parameters:
    - obj: torch.Size or tuple or list, the input object.

    Returns:
    - torch.Size, the converted torch.Size object.
    """
    if isinstance(obj, torch.Size):
        return obj
    else:
        return torch.Size(obj)
    
def pad_tensors_to_match(tuple1, tuple2):
    max_length = max(len(tuple1), len(tuple2))

    padded_tuple1 = []
    for i in range(max_length):
        tensor1 = tuple1[i] if i < len(tuple1) else torch.zeros_like(tuple2[i])
        tensor2 = tuple2[i] if i < len(tuple2) else torch.zeros_like(tuple1[i])

        # Add a second dimension if tensor1 or tensor2 is 1-dimensional
        tensor1 = tensor1.unsqueeze(1) if tensor1.ndimension() == 1 else tensor1
        tensor2 = tensor2.unsqueeze(1) if tensor2.ndimension() == 1 else tensor2

        # Determine the maximum size along each axis
        max_size = torch.tensor([max(tensor1.size(0), tensor2.size(0)),
                                 max(tensor1.size(1), tensor2.size(1))])

        # Pad tensors to match the size of the larger tensor along the first dimension (axis 0)
        padded_tensor1 = F.pad(tensor1, (0, 0, 0, max_size[0] - tensor1.size(0)))
        padded_tuple1.append(padded_tensor1)

    return tuple(padded_tuple1), tuple(tuple2)



def repeat_tensor(input_tensor, target_size):
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")
    
    if not isinstance(target_size, int) or target_size <= 0:
        raise ValueError("Target size must be a positive integer.")
    
    current_size = input_tensor.size(0)
    repetitions = target_size // current_size
    
    if repetitions * current_size < target_size:
        # If the repetitions don't fill the target size, add one more repetition
        repetitions += 1
    
    repeated_tensor = input_tensor.repeat(repetitions)[:target_size]
    return repeated_tensor

def align_tensor_dimensions(tensor1, tensor2):
    # Get the dimensions of each tensor
    dims1 = len(tensor1.shape)
    dims2 = len(tensor2.shape)

    # Add or remove dimensions in tensor2 to match tensor1
    if dims1 < dims2:
        # Add required dimensions to tensor2
        added_dims = dims2 - dims1
        tensor2 = tensor2.unsqueeze(dim=-1).expand(*tensor2.shape, added_dims)
    elif dims1 > dims2:
        # Remove extra dimensions from tensor2
        tensor2 = tensor2.squeeze(dim=-1)

    # Check if corresponding dimensions have the same length
    for dim1, dim2 in zip(tensor1.shape, tensor2.shape):
        # Determine the dimensions after elongating
        new_dim1 = max(dim1, dim2)
        new_dim2 = max(dim1, dim2)

        # Elongate the smaller dimension by concatenating sqrt(2)
        if dim1 < dim2:
            sqrt2_to_concat = torch.sqrt(torch.tensor(2)) * torch.ones(*tensor1.shape[:-1], dim2 - dim1)
            tensor1 = torch.cat([tensor1, sqrt2_to_concat], dim=-1)
        elif dim1 > dim2:
            # Remove extra dimensions from tensor2
            tensor2 = tensor2.unsqueeze(dim=-1).expand(*tensor2.shape, dim1 - dim2)
            tensor2 = tensor2.squeeze(dim=-1)

    # If all checks passed, tensors have the same dimensions
    return tensor1, tensor2

def multiply_tensors_cross(tensor1, tensor2):
    # Ensure tensors have the same size along dimension 0
    if tensor1.size(0) > tensor2.size(0):
        # Pad tensor2 with zeros along dimension 0
        padding_size = tensor1.size(0) - tensor2.size(0)
        tensor2 = torch.cat([tensor2, torch.zeros(padding_size, *tensor2.shape[1:])], dim=0)

    elif tensor1.size(0) < tensor2.size(0):
        # Repeat tensor1 to match the size of tensor2 along dimension 0
        repeat_factor = tensor2.size(0) // tensor1.size(0)
        if 1 <= len(tensor1.size()):
            result = tensor1 * tensor1
        else:    
            tensor1 = tensor1.repeat(repeat_factor, 1)

    # Ensure tensors have the same size along dimension 1
    if tensor1.size(1) > tensor2.size(1):
        # Pad tensor2 with zeros along dimension 1
        padding_size = tensor1.size(1) - tensor2.size(1)
        tensor2 = torch.cat([tensor2, torch.zeros(*tensor2.shape[0:1], padding_size, *tensor2.shape[2:])], dim=1)

    elif tensor1.size(1) < tensor2.size(1):
        # Repeat tensor1 to match the size of tensor2 along dimension 1
        repeat_factor = tensor2.size(1) // tensor1.size(1)
        if 1 <= len(tensor1.size()):
            result = tensor1 * tensor1
        else:    
            tensor1 = tensor1.repeat(repeat_factor, 1)
    if tensor1.size() != tensor2.size():
        result = tensor1 * tensor1
    # Now, tensors should have the same size along dimensions 0 and 1
    else:
        result = tensor1 * tensor2
    
    return result


def pad_to_match_size(target_tensor, tensor_to_pad):
    # # Calculate the padding needed for each dimension
    # if target_tensor.dim() != tensor_to_pad.dim():
    #         raise ValueError("Tensors must have the same number of dimensions.")

        # Calculate the padding needed for the last dimension
    last_dim_padding = max(0, target_tensor.shape[-1] - tensor_to_pad.shape[-1])

        # Pad the tensor along the last dimension using F.pad
    padding = [math.sqrt(2)] * (len(tensor_to_pad.shape) - 1) + [0, last_dim_padding]
    padded_tensor = F.pad(tensor_to_pad, padding)

    return padded_tensor


def add_tensors_cross(tensor1, tensor2):
    result = 0
    if tensor1.size(1) > tensor2.size(0):
        # Pad tensor2 with zeros along dimension 0
        padding_size = tensor1.size(0) - tensor2.size(0)
        tensor2 = torch.cat([tensor2, torch.zeros(padding_size, *tensor2.shape[1:])], dim=0)
        result = tensor1 + tensor2

    elif tensor1.size(1) < tensor2.size(0):
        # Repeat tensor1 to match the size of tensor2 along dimension 0
        repeat_factor = tensor2.size(0) // tensor1.size(0)
        tensor1 = tensor1.repeat(repeat_factor, 1)
        result = tensor1 + tensor2
    # Now, tensors should have the same size along dimension 0
    return result

import torch

def repeat_and_pow(tensor1, tensor2):
    """
    Repeat tensor2 to match the length of tensor1 and raise tensor1 to the power of tensor2.

    Parameters:
    - tensor1: torch.Tensor, the base tensor.
    - tensor2: torch.Tensor, the tensor specifying the powers.

    Returns:
    - result_tensor: torch.Tensor, the result of raising tensor1 to the power of tensor2.
    """

    # Ensure both tensors have the same number of dimensions
    if tensor1.dim() != tensor2.dim():
        raise ValueError("Both tensors should have the same number of dimensions.")

    # Determine the axis along which to repeat tensor2
    repeat_axis = tensor1.dim() - 1

    # Determine the number of times to repeat tensor2 along the specified axis
    repeat_times = tensor1.size(repeat_axis)

    # Repeat tensor2 to match the length of tensor1 along the specified axis
    repeated_tensor2 = tensor2.unsqueeze(repeat_axis).repeat(1, repeat_times)

    # Raise tensor1 to the power of repeated_tensor2
    result_tensor = torch.pow(tensor1, repeated_tensor2)

    return result_tensor

def add_or_transpose(tensor1, tensor2):
    try:
        # Attempt to add tensors directly
        result = tensor1 + tensor2
    except RuntimeError as e:
        if "The size of tensor a" in str(e):
            # Incompatible shapes, transpose tensor2 and try again
            result = tensor1 + tensor2.t()
        else:
            # Unexpected error, raise it
            raise e

    return result




        

# Custom Forward Functions

def custom_square_unweighted(input, bias=None):
    # Check if bias is provided
    if bias is not None:
        # Perform the linear transformation with bias
        
        return torch.sqrt(2*input) + reshape_tensor_to_match_one(bias,input)
    else:
        # Perform the linear transformation without bias
        return torch.sqrt(2*input)

def custom_square_weighted(input, weight, bias=None):
    # weights must be square 
    
    # Check if bias is provided
    max_value_array_input = replicate_max_value(input)
    max_value_array_weight = replicate_max_value(weight)
    input_calc = max_value_array_input/2 - input
    weight_calc = max_value_array_weight/2 - weight
    square_weight = truncate_to_square(weight_calc)
    square_weight = repeat_and_pad_to_shape(square_weight, input_calc.shape)
    if bias is not None:
        # Perform the linear transformation with bias
        mult_calc = matmul_or_transpose(input_calc, square_weight)
        pad_shape = max_value_array_input.shape
        attr = pad_to_shape(mult_calc, pad_shape)
        sub_calc = subtract_tensors(max_value_array_input,attr)
        sub_shape = sub_calc.shape
        
        bias = pad_1d_tensor(bias, sub_shape[1])


        return torch.sqrt(sub_calc) + bias
    else:
        # Perform the linear transformation without bias
        return torch.sqrt(max_value_array_input - torch.matmul((max_value_array_input/2 - input),torch.inverse((max_value_array_weight/2 - weight))))
    
def custom_cross(input, weight, bias = None):
    # chunk it 


    if bias is not None:
        # Perform the linear transformation with bias
        # weight = torch.transpose(weight, 0,1)
        base = multiply_tensors_cross(input, weight)
        bias = repeat_tensor(bias, base.shape[1])
        bias = pad_to_match_size(base, bias)
        return  add_or_transpose(base, bias)
    else:
        # Perform the linear transformation without bias
        return input * weight.t()
        
def custom_angle(input, weight, bias = None):
    
    if bias is not None:  
        return calculate_angle(input, weight)
    else:
        
        return calculate_angle(input, weight) + bias
        
        
def custom_quadratic_one(input, weight, input_pow, bias = None):
    base = input * input
    base = base.to(dtype=torch.float32)
    weight = weight.to(dtype=torch.float32)
    input = input.to(dtype=torch.float32)
    if bias is not None:
        # Perform the linear transformation with bias
        return matmul_or_transpose(base, weight.t()) + matmul_or_transpose(input, weight.t()) + bias
    else:
 
        # Perform the linear transformation without bias
        return matmul_or_transpose(base, weight) + matmul_or_transpose(input, weight)
    

def custom_quadratic_two(input, weight, input_pow, weight_pow, bias = None):
    base = input * input
    base = base.to(dtype=torch.float32)
    weight = weight.to(dtype=torch.float32)
    input = input.to(dtype=torch.float32)
    if bias is not None:
        # Perform the linear transformation with bias
        return matmul_or_transpose(base, weight.t()) + bias
    else:
        # Perform the linear transformation without bias
        return matmul_or_transpose(base, weight.t())
    
    
########################################################################################


# Custom Neural Network Layers 
class CustomLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinearLayer, self).__init__()

        # Define the learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)

        # Initialize biases to zeros if they exist
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform the linear transformation
        out = torch.nn.functional.linear(x.float(), self.weight.float(), self.bias)
        return out
    
    
class CustomSquareUnweightedLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomSquareUnweightedLayer, self).__init__()

        # Define the learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)

        # Initialize biases to zeros if they exist
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform the linear transformation
        out = custom_square_unweighted(x, self.bias)
        return out
    
class CustomSquareWeightedLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomSquareWeightedLayer, self).__init__()

        # Define the learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)

        # Initialize biases to zeros if they exist
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform the linear transformation
        out = custom_square_weighted(x, self.weight, self.bias)
        return out
    
    
class CustomCrossLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomCrossLayer, self).__init__()

        # Define the learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)

        # Initialize biases to zeros if they exist
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform the linear transformation
        out = custom_cross(x, self.weight, self.bias)
        return out
    
class CustomAngleLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomAngleLayer, self).__init__()

        # Define the learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)

        # Initialize biases to zeros if they exist
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform the linear transformation
        out = custom_angle(x, self.weight, self.bias)
        return out
    
    
class CustomQuadraticOneLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomQuadraticOneLayer, self).__init__()

        # Define the learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.input_pow = nn.Parameter(torch.Tensor(in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)

        # Initialize biases to zeros if they exist
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform the linear transformation
        out = custom_quadratic_one(x, self.weight, self.input_pow, self.bias)
        return out

class CustomQuadraticTwoLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomQuadraticTwoLayer, self).__init__()

        # Define the learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.input_pow = nn.Parameter(torch.Tensor(out_features))
        self.weight_pow = nn.Parameter(torch.Tensor(out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)

        # Initialize biases to zeros if they exist
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform the linear transformation
        out = custom_quadratic_two(x, self.weight, self.input_pow, self.weight_pow, self.bias)
        return out
    
################################################################################

#CUSTOM MODELS
# Define a simple neural network with two different dense layers
class CustomLinearModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(CustomLinearModel, self).__init__()
        self.layer1 = CustomLinearLayer(input_size, hidden_size1)
        self.layer2 = CustomLinearLayer(hidden_size1, hidden_size2)
        self.output_layer = CustomLinearLayer(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

class CustomSquareUnweightedModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(CustomSquareUnweightedModel, self).__init__()
        self.layer1 = CustomSquareUnweightedLayer(input_size, hidden_size1)
        self.layer2 = CustomSquareUnweightedLayer(hidden_size1, hidden_size2)
        self.output_layer = CustomSquareUnweightedLayer(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x
    
class CustomSquareWeightedModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(CustomSquareWeightedModel, self).__init__()
        self.layer1 = CustomSquareWeightedLayer(input_size, hidden_size1)
        self.layer2 = CustomSquareWeightedLayer(hidden_size1, hidden_size2)
        self.output_layer = CustomSquareWeightedLayer(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x
    
class CustomCrossModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(CustomCrossModel, self).__init__()
        self.layer1 = CustomCrossLayer(input_size, hidden_size1)
        self.layer2 = CustomCrossLayer(hidden_size1, hidden_size2)
        self.output_layer = CustomCrossLayer(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.leaky_relu = torch.nn.LeakyReLU(0.01)
        self.prelu = torch.nn.PReLU()
    
    def are_all_tensors(*tup):
      
        tup = tup[1]
        def convert_element(elem):
            if torch.is_tensor(elem):
                return elem
            elif isinstance(elem, CustomCrossModel):
            # Handle CustomCrossModel instances
            # You might need to implement a specific conversion logic for this type
                return elem.data  # Replace ... with appropriate conversion
            else:
                return torch.tensor(elem)

        all_tensors = [convert_element(elem) for elem in tup]
        return all(torch.is_tensor(elem) for elem in all_tensors)
    
    def convert_to_tensors(*tup):
        return tuple(torch.tensor(0) if not torch.is_tensor(elem) else elem for elem in tup)
    
    def forward_tensor(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        x = F.softmax(x, dim=1)
        return x

    def forward(self, x):
    
        if isinstance(x, tuple):
            x = torch.flatten(torch.cat(x[1:]), start_dim=0)
        y = x
        x = []
        x.append(self.forward_tensor(y))
        x = tuple(x)
     
        return x



    
class CustomAngleModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(CustomAngleModel, self).__init__()
        self.layer1 = CustomAngleLayer(input_size, hidden_size1)
        self.layer2 = CustomAngleLayer(hidden_size1, hidden_size2)
        self.output_layer = CustomAngleLayer(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x
    
class CustomQuadOneModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(CustomQuadOneModel, self).__init__()
        self.layer1 = CustomQuadraticOneLayer(input_size, hidden_size1)
        self.layer2 = CustomQuadraticOneLayer(hidden_size1, hidden_size2)
        self.output_layer = CustomQuadraticOneLayer(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

class CustomQuadTwoModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(CustomQuadTwoModel, self).__init__()
        self.layer1 = CustomQuadraticTwoLayer(input_size, hidden_size1)
        self.layer2 = CustomQuadraticTwoLayer(hidden_size1, hidden_size2)
        self.output_layer = CustomQuadraticTwoLayer(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x
    
    
    
######################################################################################

# WEIGHT INITIALISATION MODELS


class CustomLinearXavierModel(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinearXavierModel, self).__init__()
        
          # Assuming you have Linear layers in your model
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)

        # Initialize the weights using Xavier initialization
        init.xavier_uniform_(self.layer1.weight)
        init.xavier_uniform_(self.layer2.weight)
        init.xavier_uniform_(self.output_layer.weight)

        # Define the learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        init_xavier_uniform(self.weight)

        # Initialize biases to zeros if they exist
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform the linear transformation
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        return out
    
class CustomLinearKaimingModel(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinearKaimingModel, self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)

        # Initialize the weights using Xavier initialization
        init.kaiming_uniform_(self.layer1.weight)
        init.kaiming_uniform_(self.layer2.weight)
        init.kaiming_uniform_(self.output_layer.weight)

        # Define the learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        init_kaiming_uniform(self.weight)

        # Initialize biases to zeros if they exist
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform the linear transformation
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        return out
    
class CustomLinearZerosModel(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinearZerosModel, self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)

        init.zeros_(self.layer1.weight)
        init.zeros_(self.layer2.weight)
        init.zeros_(self.output_layer.weight)

        
        # Define the learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        init_zeros(self.weight)

        # Initialize biases to zeros if they exist
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform the linear transformation
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        return out
    


class CustomLinearNormalModel(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinearNormalModel, self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)

        init.normal_(self.layer1.weight)
        init.normal_(self.layer2.weight)
        init.normal_(self.output_layer.weight)
        
        

        # Define the learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        init_normal(self.weight)

        # Initialize biases to zeros if they exist
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform the linear transformation
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        return out
    

class CustomLinearOrthogonalModel(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinearOrthogonalModel, self).__init__()

        
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)

        init.orthogonal_(self.layer1.weight)
        init.orthogonal_(self.layer2.weight)
        init.orthogonal_(self.output_layer.weight)
        
        # Define the learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        init_orthogonal(self.weight)

        # Initialize biases to zeros if they exist
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform the linear transformation
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        return out
    

class CustomLinearUniformModel(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinearUniformModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)

        init.uniform_(self.layer1.weight)
        init.uniform_(self.layer2.weight)
        init.uniform_(self.output_layer.weight)
        
        # Define the learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        init_uniform(self.weight)

        # Initialize biases to zeros if they exist
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform the linear transformation
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        return out
    

class CustomLinearSparseModel(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinearSparseModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)

        init.sparse_(self.layer1.weight, sparsity = 0.1)
        init.sparse_(self.layer2.weight, sparsity = 0.1)
        init.sparse_(self.output_layer.weight, sparsity = 0.1)
        # Define the learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        init_sparse(self.weight)

        # Initialize biases to zeros if they exist
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform the linear transformation
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        return out
    
    
    
################################################################################
 #Tests    
#  write the overall test - (run on 2 sentiments 1 image and then analysis - 14 models)
def train_test_sentiment(model_name, model, csv_file_path, data_name):
    import torch 
    class SentimentDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            row = self.data.iloc[index]
            sentence = row['Sentence']  # Assuming 'Sentence' is the column name
            label = row['Label'] 
            tokens = self.tokenizer(sentence.lower())
            
            if len(tokens) < self.max_length:
                tokens = tokens + [''] * (self.max_length - len(tokens))
            else:
                tokens = tokens[:self.max_length]
            tokens = [token2id.get(token, 0) for token in tokens]  # Convert tokens to ids
            padding_length = self.max_length - len(tokens)
            tokens += [0] * padding_length  # Pad with zeros if needed
            return torch.tensor(tokens), torch.tensor(label)
        
    csv_file_path = csv_file_path
    df = pd.read_csv(csv_file_path)
    
    # Encode labels as numerical values
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Sentiment'])
    
    tokenizer = lambda x: x.split()  # Example: simple space-based tokenizer
    all_tokens = [token for sentence in df['Sentence'] for token in tokenizer(sentence.lower())]
    token_counts = Counter(all_tokens)
    
    vocab_size = len(token_counts) + 1  # Add 1 for the unknown token
    token2id = {token: idx + 1 for idx, (token, _) in enumerate(token_counts.most_common())}
    
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    max_length = 50  # Choose an appropriate max length for your sentences
    train_dataset = SentimentDataset(train_data, tokenizer, max_length)
    test_dataset = SentimentDataset(test_data, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    max_length = 50  # Choose an appropriate max length for your sentences
    train_dataset = SentimentDataset(train_data, tokenizer, max_length)
    test_dataset = SentimentDataset(test_data, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []  # To store training losses for plotting
    def any_true(array):
        return any(element == True for element in array)
    
    def ensure_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj  # Already a tensor, no need to convert
        elif isinstance(obj, tuple):
            multi = []
            result = 0
            for item in obj:
                if isinstance(item, torch.Tensor) and item.numel() == 1:
                    multi.append(False)
                elif isinstance(item, torch.Tensor) and item.numel() > 1:
                    multi.append(True)
                    result += torch.sum(item)
                else:
                    raise ValueError("Tuple must contain tensors.")
            if any(multi):
                return result
            else:
                return torch.tensor(obj)
        else:
            raise ValueError("Input must be either a tensor or a tuple.")
        
    import torch


    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_inputs, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            # outputs = ensure_tensor(outputs)
            batch_labels = ensure_tensor(batch_labels)
            if isinstance(outputs, tuple) and len(outputs) == 1:
                list_obj = [torch.tensor(arr, requires_grad=True) for arr in outputs]
                outputs = torch.stack(list_obj)
                outputs = F.softmax(outputs, dim=1).squeeze()
            if isinstance(batch_labels, tuple) and len(batch_labels) == 1:
                list_obj = [torch.tensor(arr) for arr in batch_labels]
                batch_labels = torch.stack(list_obj)
                batch_labels = F.softmax(batch_labels, dim=1).squeeze()
            if isinstance(batch_labels, torch.Tensor) and batch_labels.size == 1:
                batch_labels = F.softmax(batch_labels, dim=1).squeeze()
            if len(batch_labels) != len(outputs):
              
                batch_labels = torch.arange(1, len(outputs) + 1)
                batch_labels = batch_labels.float()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        train_losses.append(average_loss)
        if np.any(train_losses):
            train_losses = np.nan_to_num(train_losses)
            train_losses = train_losses.tolist()
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}")
        
            # Save the training loss plot
    plt.plot(range(1, epochs + 1), train_losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.savefig(f'{model_name} {data_name} training_loss_plot.png')
    plt.close()

    # Evaluation on the test set
    model.eval()
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch_inputs, batch_labels in tqdm(test_loader, desc="Testing"):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            outputs = model(batch_inputs)
            outputs = torch.flatten(outputs, start_dim=1)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(batch_labels.cpu().numpy())

    # Save evaluation results to a CSV file
    evaluation_results = pd.DataFrame({
        'True Labels': all_true_labels,
        'Predictions': all_predictions
    })
    evaluation_results.to_csv(f'{model_name} {data_name} evaluation_results.csv', index=False)

    # Calculate accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
            
        
############################################################################################

#IMAGES

def train_test_images(model_name, model):

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    root_dir = 'test_train_data/Agricultural-crops'
    
    all_image_paths = []
    all_labels = [] 
    
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)

        # Check if the item in the parent directory is a subfolder
        if os.path.isdir(class_path):
            # Iterate through images in the subfolder
            for filename in os.listdir(class_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    # Append image path and label to the lists
                    image_path = os.path.join(class_path, filename)
                    all_image_paths.append(image_path)
                    all_labels.append(class_name)

    # Create a dataset from the image paths and labels
    class CustomImageDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, index):
            image_path, label = self.image_paths[index], self.labels[index]

            # Load image using torchvision.transforms
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            return image, label

    # Split the dataset into training and testing sets
    image_paths_train, image_paths_test, labels_train, labels_test = train_test_split(
        all_image_paths, all_labels, test_size=0.2, random_state=42
    )

    # Create dataloaders
    train_dataset = CustomImageDataset(image_paths_train, labels_train, transform=transform)
    test_dataset = CustomImageDataset(image_paths_test, labels_test, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    num_classes = len(set(train_dataset.labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []  # To store training losses for plotting

    label_encoder = LabelEncoder()
    
    def encode_string(string, char_to_index):
    # Convert each character to its corresponding index
        indices = [char_to_index[char] for char in string]
        return torch.tensor(indices, dtype=torch.long)
    
    def ensure_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj  # Already a tensor, no need to convert
        elif isinstance(obj, tuple):
            multi = []
            result = 0
            for item in obj:
                if isinstance(item, torch.Tensor) and item.numel() == 1:
                    multi.append(False)
                elif isinstance(item, torch.Tensor) and item.numel() > 1:
                    multi.append(True)
                    result += torch.sum(item)
                else:
                    raise ValueError("Tuple must contain tensors.")
            if any(multi):
                return result
            else:
                return torch.tensor(obj)
        else:
            raise ValueError("Input must be either a tensor or a tuple.")

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            char_to_index = {char: index for index, char in enumerate(set(labels))}
            labels = encode_string(labels, char_to_index)
            optimizer.zero_grad()
            outputs = model(inputs)
            labels_one_hot = F.one_hot(labels, num_classes)
            # #####
            inputs = ensure_tensor(inputs)
            if isinstance(outputs, tuple) and len(outputs) == 1:
                list_obj = [torch.tensor(arr, requires_grad=True) for arr in outputs]
                outputs = torch.stack(list_obj)
                outputs = F.softmax(outputs, dim=1).squeeze()
            if isinstance(inputs, tuple) and len(inputs) == 1:
                list_obj = [torch.tensor(arr) for arr in inputs]
                inputs = torch.stack(list_obj)
                inputs = F.softmax(inputs, dim=1).squeeze()
            if isinstance(inputs, torch.Tensor) and inputs.size == 1:
                inputs = F.softmax(inputs, dim=1).squeeze()
            if len(inputs) != len(outputs):
        
                inputs = torch.arange(1, len(outputs) + 1)
                inputs = inputs.float()
            #######
            loss = criterion(outputs, inputs)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
        
            
        plt.plot(range(1, len(train_losses)+1), [loss.item() for loss in train_losses], marker='o')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.savefig(f'{model_name} images training_loss_plot_epoch_{epoch + 1}.png')
        plt.close()

    # Evaluation on the test set
    model.eval()
    all_predictions = []
    all_true_labels = []
    known_indices = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            char_to_index = {char: index for index, char in enumerate(set(labels))}
            labels = encode_string(labels, char_to_index)
            outputs = model(inputs)
            _, predictions = torch.max(outputs[0], 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    def pad_shorter_array(array1, array2, pad_value=0):
        
        array1 = np.array(array1).ravel()
        array2 = np.array(array2).ravel()
        len1, len2 = len(array1), len(array2)

        if len1 < len2:
            array1 = np.pad(array1, (0, len2 - len1), constant_values=pad_value)
        elif len2 < len1:
            array2 = np.pad(array2, (0, len1 - len2), constant_values=pad_value)

        return array1, array2

    all_true_labels, all_predictions = pad_shorter_array(all_true_labels, all_predictions)
    
    evaluation_results = pd.DataFrame({
    'True Labels': all_true_labels,
    'Predictions': all_predictions
    })
    evaluation_results.to_csv(f'{model_name} evaluation_results.csv', index=False)
    # Calculate accuracy
    def convert_to_numpy_integers(lst):
        return [np.int64(value) for value in lst]
    all_true_labels = convert_to_numpy_integers(all_true_labels)
    
    def filter_multiclass_arrays(arrays):

        return [arr for arr in arrays if arr.ndim == 1]
    flat_pred = [pred.flatten() for pred in all_predictions]
    flat_pred_1 = [arr for arr in flat_pred if arr.ndim == 1]
    filtered_predicitons = filter_multiclass_arrays(flat_pred_1)
    flattened_pred = [arr.flatten() for arr in filtered_predicitons]
    flat_pred_y = np.argmax(flat_pred, axis=1)
    flat_pred_1_y = np.argmax(flat_pred_1, axis=1)
    filtered_predictions_y = np.argmax(filtered_predicitons, axis=1)
    flattened_pred_y_1 = np.argmax(flattened_pred, axis=1)
    try: 
        print("Attempt 1")
        accuracy = accuracy_score(np.array(all_true_labels), flat_pred_y)
        print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
        
    except ValueError:
        try:
            print("Attempt 2")
            accuracy = accuracy_score(np.array(all_true_labels), flat_pred_1_y)
            print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
            
        except ValueError:
            try:
                print("Attempt 3")
                accuracy = accuracy_score(np.array(all_true_labels), filtered_predictions_y)
                print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
            except ValueError:
                try:
                    print("Attempt 4")
                    accuracy = accuracy_score(np.array(all_true_labels), flattened_pred_y_1)
                    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
                except ValueError:
                    print("No cases available")
                    
                    
                
            
            
            
        
    


#  implement it for an image set 

# 

# Function to visualize model architecture
def visualize_model_architecture(model):
    print(model)

# Function to visualize activations
def visualize_activations(model, input_data):
    activations = []

    def hook_fn(module, input, output):
        current_activation = output.detach().numpy()
        print(f'Activation shape for layer {len(activations) + 1}: {current_activation.shape}')
        activations.append(current_activation)

    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    model(input_data)

    for hook in hooks:
        hook.remove()

    # Plot activations
    plt.figure(figsize=(12, 6))
    for i, activation in enumerate(activations):
        plt.subplot(1, len(activations), i + 1)
        plt.title(f'Layer {i + 1} Activation')

        # Check if the activation is 1D or 2D
        if activation.ndim == 1:
            plt.plot(activation)
        else:
            plt.imshow(activation[0].reshape(len(activation[0]),1), cmap='viridis')
            plt.colorbar()

    plt.tight_layout()
    plt.show()



# Function for weight and bias analysis
def analyze_weights_biases(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            plt.figure(figsize=(6, 4))
            plt.title(f'{name} Distribution')
            plt.hist(param.data.numpy().flatten(), bins=50, color='blue', alpha=0.7)
            plt.show()

# Function for model ablation study

def pad_tensors(tensors):
    # Find the maximum size along each dimension
    if isinstance(tensors, torch.Tensor):
        max_sizes = torch.tensor([max(tensor.size(dim) for tensor in tensors) for dim in range(tensors[0].dim())])
        padded_tensors = [pad_tensor(tensor, max_sizes) for tensor in tensors]
    elif isinstance(tensors, list):
        max_sizes = max([max(len(tensor) for tensor in tensors)])
        padded_tensors = [pad_tensor(tensor, max_sizes) for tensor in tensors]
        

    # Pad each tensor to match the maximum size along each dimension
    

    return padded_tensors

def flatten_list(input_list):
  """
  This function takes a list and flattens any sublists within it.

  Args:
    input_list: A list of elements, some of which may be sublists.

  Returns:
    A new list containing all the elements from the input list,
    in order, with any sublists flattened.
  """
  flat_list = []
  for element in input_list:
    if isinstance(element, list):
      flat_list.extend(flatten_list(element))
    else:
      flat_list.append(element)
  return flat_list

def pad_list(list_arr):
    # list_arr = [list_arr]
    list_arr = [item for sublist in list_arr for item in sublist]
    # list_arr = [list_arr.tolist()]
    max_len = max([max(len(arr) for arr in list_arr)])
    count_arrs = len(list_arr)
    arr_padded = [0] * count_arrs
    pad_length = [0] * count_arrs
    padding = [0] * count_arrs
    if count_arrs == 0:
        return None
    else:
        for i in range(0, count_arrs):
            pad_length[i] = max_len - len(list_arr[i])
            if pad_length[i] == 0:
                continue
            # padding[i] = list(map(list, zip(*[[0] * pad_length[i]])))
            list_arr[i] = np.concatenate((list_arr[i], [0] * pad_length[i]), axis=0)
        
        list_arr = torch.tensor(list_arr)
        list_arr = torch.squeeze(list_arr, dim=0)
        num = list_arr.numel()
        num = num/(125*5)
        print(num)
        list_arr = list_arr.view(int(num),125,5)
      
                
    
    return list_arr


def pad_tensor(tensor, target_sizes):
    # Calculate the padding required for each dimension
    if isinstance(tensor, torch.Tensor):
        padding = [0, 0] * tensor.dim()  # Initialize padding for each dimension
        for dim, target_size in enumerate(target_sizes):
            current_size = tensor.size(dim)
            if current_size < target_size:
                padding[2 * dim + 1] = target_size - current_size  # Set padding for the upper side
                
        padded_tensor = torch.nn.functional.pad(tensor, padding)

    if isinstance(tensor, list):
        # padding = [0, target_sizes - len(tensor)]
        # tensor = sum([sublist for sublist in tensor])
        # tensor_array = np.array(tensor)
        # padded_tensor_array = np.pad(tensor_array, padding, mode='constant', constant_values=0)
        # padded_tensor = padded_tensor_array.tolist()
        # tensor = torch.tensor(tensor)
        padded_tensor = pad_list(tensor)
    # Apply padding to the tensor

    return padded_tensor


        

def compare_ablation_to_target(ablation_result):
        # Convert ablation result to torch tensor
        
        ablation_tensor = pad_tensors(ablation_result)
        # print(type(ablation_tensor))
        # if not isinstance(ablation_tensor, torch.Tensor):
        #     ablation_tensor = torch.tensor(ablation_tensor)
            

        # Calculate percentage difference
        # print(target_data.shape)
        # print(ablation_tensor.shape)
        
        

        return ablation_tensor
    
def plot_percentage_difference(ablation_tensor, target_data, filename):
        
        
       
        labels = ["Input", "Layer 1", "Layer 2", "Output", "ReLU"]
        print(f'The length of the ablation tensor is {len(ablation_tensor)}')
        for i in range(0, len(ablation_tensor)):
            abla = ablation_tensor[i][:, :100][0]
            percentage_difference = ((target_data - abla) / target_data) * 100
            per = percentage_difference.squeeze().numpy()
            plt.plot(per, marker='o', label=labels)
            plt.title('Percentage Difference')
            plt.xlabel('Output Dimension')
            plt.ylabel('Percentage Difference')
            plt.savefig(f'Number {i} ablation study percentage difference {filename}')  # Save the plot as a PNG
            plt.legend()
            plt.show()
    
def ablation_study(model, input_data):
    ablated_outputs = []

    def hook_fn(module, input, output):
        ablated_outputs.append(output.detach().numpy())

    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Ablate the second layer
    model.layer2.weight.requires_grad = False

    model(input_data)

    for hook in hooks:
        hook.remove()
        
    
        
    

# Function to plot the percentage difference
    

        return ablated_outputs

# Function for runtime and memory analysis
def analyze_runtime_memory(model, input_data):
    import time

    start_time = time.time()
    model(input_data)
    end_time = time.time()

    runtime = end_time - start_time

    print(f"Runtime: {runtime:.4f} seconds")

    # Memory analysis (PyTorch does not provide direct memory usage information)

# Function for hyperparameter tuning
def hyperparameter_tuning(model, input_data, target_data):
    learning_rates = [0.001, 0.01, 0.1]
    losses = []

    for lr in learning_rates:
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(10):
            outputs = model(input_data)
            loss = criterion(outputs, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    plt.figure(figsize=(8, 5))
    plt.plot(learning_rates, losses, marker='o')
    plt.title('Hyperparameter Tuning: Learning Rate vs. Loss')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()
    
    
    # Function to visualize model architecture and save as PNG
def visualize_model_architecture_and_save(model, filename):
    model_str = str(model)
    with open(filename + '.txt', 'w') as f:
        f.write(model_str)


def convert_tensors_to_numpy(tensors_tuple):
    """
    Convert each tensor in a tuple to a NumPy array after detaching.

    Parameters:
    - tensors_tuple (tuple): Tuple of PyTorch tensors.

    Returns:
    - numpy_arrays (list): List of NumPy arrays.
    """
    numpy_arrays = [torch.tensor(tensor).detach().numpy() for tensor in tensors_tuple]
    return numpy_arrays

# Function to visualize activations and save as PNG
def visualize_activations_and_save(model, input_data, filename):
    activations = []
    
    def calculate_activation_shape(array_size, target_columns=1):
    # Check if the total number of elements is divisible by the target_columns
        if array_size % target_columns != 0:
            raise ValueError("Cannot evenly reshape array with size {} into {} columns.".format(array_size, target_columns))

        # Calculate the number of rows based on the target_columns
        target_rows = array_size // target_columns

        return target_rows, target_columns

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations.append(convert_tensors_to_numpy(output))
        else:
            current_activation = output.detach().numpy()
            activations.append(current_activation)

    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    model(input_data)

    for hook in hooks:
        hook.remove()

    plt.figure(figsize=(12, 6))
    for i, activation in enumerate(activations):
        plt.subplot(1, len(activations), i + 1)
        plt.title(f'Layer {i + 1} Activation')

        if isinstance(activation, np.ndarray) and activation.size > 0:
            
    
    # Check if the array is 0-dimensional (scalar)
            if activation.ndim == 0:
                # Handle the scalar case as needed
                value = activation.item()
                print("Scalar value:", value)
            else:
                activation = activation.ravel()
                activation = activation.reshape(-1,1)
                plt.imshow(activation, cmap='viridis')
        else:
            print("Activation array is empty or not a valid numpy array.")

    plt.tight_layout()
    plt.savefig(filename + '_activations.png')
    plt.close()

# Function for weight and bias analysis and save as PNG
def analyze_weights_biases_and_save(model, filename):
    for name, param in model.named_parameters():
        if 'weight' in name:
            plt.figure(figsize=(6, 4))
            plt.title(f'{name} Distribution')
            data = param.data.numpy().flatten()
            nan_indices = np.isnan(data)
            if np.any(nan_indices):
                data = np.nan_to_num(data)
    # Handle NaN values (e.g., replace them with a specific value or remove them)
            plt.hist(param.data.numpy().flatten(), bins=50, color='blue', alpha=0.7)
            plt.savefig(filename + f'_{name}_distribution.png')
            plt.close()
def pad_2d_lists(input_2d_lists, padding_value=0):
    # Find the maximum number of rows among all 2D lists
    max_rows = max(len(lst) for lst in input_2d_lists)

    # Pad shorter rows in each 2D list with the specified padding value
    padded_2d_lists = [[row + [padding_value] * (max_rows - len(row)) for row in lst] for lst in input_2d_lists]

    return padded_2d_lists


def get_lengths_at_dimension(lst, dimension):
    lst = [lst]
    if dimension == 0:
        return len(lst)
    elif dimension > 0:
        return [get_lengths_at_dimension(inner_list, dimension - 1) for inner_list in lst]
    else:
        raise ValueError("Dimension must be a non-negative integer.")
    
def pad_arrays(ablated_outputs):
    """
    Pad arrays within ablated_outputs to ensure consistent length.

    Parameters:
    - ablated_outputs (list): List of arrays.

    Returns:
    - padded_outputs (list): List of padded arrays
    """
    if isinstance(ablated_outputs, tuple):
       
        if len(ablated_outputs) == 0:
            return ablated_outputs
 
    # Ensure all inner lists have the same length
    def has_len(obj):
        return hasattr(obj, "__len__")
    
    if has_len(ablated_outputs) == False:
        return ablated_outputs
    
    if isinstance(ablated_outputs, list):
        outputs = []
        if has_len(ablated_outputs) == False:
            return ablated_outputs
        if isinstance(ablated_outputs, np.ndarray):
            
            ablated_outputs = ablated_outputs.flatten()
        ablated_outputs = [ablated_outputs]
        lengths_dimension_1 = [len(inner_list) for inner_list in ablated_outputs]
        lengths_dimension_2 = get_lengths_at_dimension(ablated_outputs, 0)            
        max_length = max(len(inner_list) for inner_list in ablated_outputs)
        for i in range(0, len(lengths_dimension_1)):
            if isinstance(lengths_dimension_2, list):
                for j in range(0, len(lengths_dimension_2)):
                    outputs.append(np.pad(ablated_outputs[i][j], (0, max_length - lengths_dimension_2[i][j])))
            elif isinstance(lengths_dimension_2, int):
                return ablated_outputs
                
    elif len(ablated_outputs) == 0:
        return ablated_outputs
            
    # Convert to NumPy array
    
    ablated_outputs = np.vstack(outputs)
    
    
    if isinstance(ablated_outputs, np.ndarray):
        if ablated_outputs.ndim == 0:
            return np.atleast_1d(ablated_outputs)
        

    # Find the maximum length among all arrays

    target_length = max(
        max(array.shape[0] if isinstance(array, np.ndarray) and array.ndim > 0 else 0 for array in inner_list)
        for inner_list in ablated_outputs
    )

    
    padded_outputs = [
        np.pad(array, ((0, max(0, target_length - (array.shape[0] if isinstance(array, np.ndarray) and array.ndim > 0 else 0))), (0, 0)), mode='constant')
        if isinstance(array, np.ndarray) and array.ndim > 0
        else array
        for inner_list in ablated_outputs
        for array in inner_list
    ]
    
    # padded_outputs = []

    # for inner_list in ablated_outputs:
    #     padded_inner_list = []

    #     for array in inner_list:
    #         current_length = array.shape[0]

    #         if current_length < target_length:
    #             # Pad the array if its length is less than the target length
    #             padding_length = target_length - current_length
    #             padded_array = np.pad(array, (0, padding_length), mode='constant')
    #             padded_inner_list.append(padded_array)
    #         else:
    #             # If the length is equal or greater than the target, no padding is needed
    #             padded_inner_list.append(array)

    #     padded_outputs.append(padded_inner_list)
    if isinstance(padded_outputs, np.ndarray):
        
        padded_outputs.tolist()
        
    return padded_outputs

# Function for model ablation study and save results as CSV
def ablation_study_and_save(model, input_data, target_data, filename):
    ablated_outputs = ablation_study(model, input_data)
    ablated_outputs = pad_arrays(ablated_outputs)
    ablation_tensor = compare_ablation_to_target(ablated_outputs)
    plot_percentage_difference(ablation_tensor, target_data, filename)
    
    
    
    # Create a DataFrame
    ablation_df = pd.DataFrame(ablated_outputs)
    ablation_df.to_csv(filename + '_ablation_results.csv', index=False)


# Function for hyperparameter tuning and save plot as PNG
def hyperparameter_tuning_and_save(model, input_data, target_data, filename):
    learning_rates = [0.001, 0.01, 0.1]
    losses = []

    
    def match_size_first_dimension(tensor1, tensor2, padding_value=0):
        if tensor1.size(0) == tensor2.size(0):
            # Tensors already have the same size along the first dimension
            return tensor1, tensor2
        elif tensor1.size(0) < tensor2.size(0):
            # Extend tensor1 along the first dimension
            padding = torch.full((tensor2.size(0) - tensor1.size(0), *tensor1.shape[1:]), padding_value)
            tensor1 = torch.cat((tensor1, padding), dim=0)
            return tensor1, tensor2
        else:
            # Extend tensor2 along the first dimension
            padding = torch.full((tensor1.size(0) - tensor2.size(0), *tensor2.shape[1:]), padding_value)
            tensor2 = torch.cat((tensor2, padding), dim=0)
            return tensor1, tensor2
    
    
    for lr in learning_rates:
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(10):
            outputs = model(input_data)
            if isinstance(outputs, tuple):
                outputs = torch.cat(outputs, dim=0)
            outputs_aligned, target_data_aligned = align_tensors(outputs, target_data)
            if target_data.shape != target_data_aligned.shape:
                target_data = target_data_aligned
            if outputs.shape != outputs_aligned.shape:
                outputs = outputs_aligned
                
            outputs, target_data = match_size_first_dimension(outputs, target_data)
 
            if (len(target_data.shape) > len(outputs.shape)):
                target_data = torch.sum(target_data, dim=1)
            loss = criterion(outputs, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    plt.figure(figsize=(8, 5))
    plt.plot(learning_rates, losses, marker='o')
    plt.title('Hyperparameter Tuning: Learning Rate vs. Loss')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.savefig(filename + '_hyperparameter_tuning.png')
    plt.close()

# Save outputs based on context


# Set random seed for reproducibility
torch.manual_seed(42)


def complete_analysis(model_name, model, input_data, target_data, filename):
    # pass through the 3 tests 
    
    # visualize_model_architecture_and_save(model, f'{model_name} model_architecture')
    # visualize_activations_and_save(model, input_data, f'{model_name} activations')
    # analyze_weights_biases_and_save(model, f'{model_name} weights_biases')
    ablation_study_and_save(model, input_data, target_data, f'{model_name} ablation_study')
    # hyperparameter_tuning_and_save(model, input_data, target_data, f'{model_name} hyperparameter_tuning')
    



# Define parameters for analyis
input_size = 50
hidden_size1 = 20
hidden_size2 = 15
output_size = 5
num_epochs = 10
learning_rate = 0.01
# Create synthetic data
input_data = torch.randn(100, input_size)
target_data = torch.randn(100, output_size)
# Create a model for analysis


## Linear Model - 1
model_name = 'Custom Linear Model'
model = CustomLinearModel(input_size, hidden_size1, hidden_size2, output_size)
data_name = 'Finance'
filename = model_name + data_name
complete_analysis('Custom Linear Model', model, input_data, target_data, filename)
# train_test_sentiment(model_name, model, 'test_train_data/finance_sentiment.csv', 'finance')
# model = CustomLinearModel(64, hidden_size1, hidden_size2, 64)
# data_name = 'Images'
# train_test_images(model_name, model)
# model = CustomLinearModel(input_size, hidden_size1, hidden_size2, output_size)
# complete_analysis('Custom Linear Model', model, input_data, target_data, data_name)



# ## Square Unweighted - 2
model_name = 'Custom Square Unweighted Model'
model = CustomSquareUnweightedModel(input_size, hidden_size1, hidden_size2, output_size)
data_name = 'Finance'
filename = model_name + data_name
complete_analysis('Custom Square Unweighted Model', model, input_data, target_data, data_name)
# train_test_sentiment(model_name, model, 'test_train_data/finance_sentiment.csv', 'finance')
# complete_analysis('Custom Square Unweighted Model', model, input_data, target_data, filename)

# model = CustomSquareUnweightedModel(64, hidden_size1, hidden_size2, 64)
# data_name = 'Images'
# train_test_images(model_name, model)
# complete_analysis('Custom Square Unweighted Model', model, input_data, target_data, data_name)

# ## Square Weighted - 3
model_name = 'Custom Square Weighted Model'
model = CustomSquareWeightedModel(input_size, hidden_size1, hidden_size2, output_size)
data_name = 'Finance'
filename = model_name + data_name
complete_analysis('Custom Square Weighted Model', model, input_data, target_data, filename)
# complete_analysis('Custom Square Weighted Model', model, input_data, target_data, data_name)
# train_test_sentiment(model_name, model, 'test_train_data/finance_sentiment.csv', 'finance')
# model = CustomSquareUnweightedModel(64, hidden_size1, hidden_size2, 64)
# data_name = 'Images'
# train_test_images(model_name, model)
# complete_analysis('Custom Square Weighted Model', model, input_data, target_data, data_name)


# ## Custom Cross - 4
# model_name = 'Custom Cross Model'
model = CustomCrossModel(input_size, hidden_size1, hidden_size2, output_size)
data_name = 'Finance'
filename = model_name + data_name
complete_analysis('Custom Cross Model', model, input_data, target_data, filename)
# train_test_sentiment(model_name, model, 'test_train_data/finance_sentiment.csv', 'finance')
# model = CustomCrossModel(64, hidden_size1, hidden_size2, 64)
# data_name = 'Images'
# train_test_images(model_name, model)
# complete_analysis('Custom Cross Model', model, input_data, target_data, data_name)


## Custom Angle - 5
model_name = "Custom Angle Model"
model = CustomAngleModel(input_size, hidden_size1, hidden_size2, output_size)
data_name = 'Finance'
filename = model_name + data_name
complete_analysis('Custom Angle Model', model, input_data, target_data, filename)
# train_test_sentiment(model_name, model, 'test_train_data/finance_sentiment.csv', 'finance')
# model = CustomAngleModel(64, hidden_size1, hidden_size2, 64)
# data_name = 'Images'
# train_test_images(model_name, model)
# complete_analysis('Custom Angle Model', model, input_data, target_data, data_name)

# # Custom Quad 1 - 6
model_name ='Custom Quad 1'
model = CustomQuadOneModel(input_size, hidden_size1, hidden_size2, output_size)
data_name = 'Finance'
filename = model_name + data_name
complete_analysis('Custom Quad One Model', model, input_data, target_data, filename)
train_test_sentiment(model_name, model, 'test_train_data/finance_sentiment.csv', 'finance')
# model = CustomQuadOneModel(64, 64, 64, 64)
# data_name = 'Images'
# train_test_images(model_name, model)
# complete_analysis('Custom Quad One Model', model, input_data, target_data, data_name)


# Custom Quad 2 - 7
model_name = 'Custom Quad 2'
model = CustomQuadTwoModel(input_size, hidden_size1, hidden_size2, output_size)
data_name = 'Finance'
filename = model_name + data_name
complete_analysis('Custom Quad Two Model', model, input_data, target_data, filename)
# train_test_sentiment(model_name, model, 'test_train_data/finance_sentiment.csv', 'finance')
# model = CustomQuadTwoModel(64, 64, 64, 64)
# data_name = 'Images'
# train_test_images(model_name, model)
# complete_analysis('Custom Quad Two Model', model, input_data, target_data, data_name)

# data_name = "Random Generated"
# model = CustomLinearXavierModel(input_size, hidden_size1, output_size)
# complete_analysis('Custom Linear Xavier Model', model, input_data, target_data, data_name)
# model = CustomLinearKaimingModel(input_size, hidden_size1, output_size)
# complete_analysis('Custom Linear Kaiming Model', model, input_data, target_data, data_name)
# model = CustomLinearZerosModel(input_size, hidden_size1, output_size)
# complete_analysis('Custom Linear Zeros Model', model, input_data, target_data, data_name)
# model = CustomLinearNormalModel(input_size, hidden_size1, output_size)
# complete_analysis('Custom Linear Normal Model', model, input_data, target_data, data_name)
# model = CustomLinearOrthogonalModel(input_size, hidden_size1, output_size)
# complete_analysis('Custom Linear Orthogonal Model', model, input_data, target_data, data_name)
# model = CustomLinearUniformModel(input_size, hidden_size1, output_size)
# complete_analysis('Custom Linear Uniform Model', model, input_data, target_data, data_name)
# model = CustomLinearSparseModel(input_size, hidden_size1, output_size)
# complete_analysis('Custom Linear Sparse Model', model, input_data, target_data, data_name)



