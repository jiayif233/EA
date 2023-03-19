# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 11:09:59 2022

@author: F-J-Y
"""

import torch
import struct
import numpy
import bitstring 

def bin2float(b):
    ''' Convert binary string to a float.

    Attributes:
        :b: Binary string to transform.
    '''
    h = int(b, 2).to_bytes(8, byteorder="big")
    return struct.unpack('>d', h)[0]


def float2bin(f):
    ''' Convert float to 64-bit binary string.

    Attributes:
        :f: Float number to transform.
    '''
    [d] = struct.unpack(">Q", struct.pack(">d", f))
    return f'{d:064b}'
'''IEEE 754 representation 
'''



def number_of_ones(n):
      one_count = 0
      for i in n:
         if i == "1":
            one_count+=1
      return one_count

def counter_ones(out):
    array_out= out.cpu().detach().numpy()
    array_out_tobytes= array_out.tobytes()
    hex_array_out_tobytes = bitstring.BitArray(array_out_tobytes)
    hex_array_out_tobytes_in_binary = hex_array_out_tobytes.bin
    ones_out2 = number_of_ones(hex_array_out_tobytes_in_binary)
    # number of ones: ones_out2
    efficiency2 = ones_out2 / len(hex_array_out_tobytes_in_binary)
    # ones / the total bits: efficiency
    return efficiency2


def counter_ones_for_params(out):
    random_tensor = torch.randn(len(out))
    for i in range(len(out)):
        random_tensor[i] = counter_ones(out[i])
    return random_tensor


def counter_ones_version_2(out):
    array_out= out.cpu().detach().numpy()
    # array_out= out.detach().numpy()
    array_out_tobytes= array_out.tobytes()
    hex_array_out_tobytes = bitstring.BitArray(array_out_tobytes)
    hex_array_out_tobytes_in_binary = hex_array_out_tobytes.bin
    ones_out2 = number_of_ones(hex_array_out_tobytes_in_binary)
    # number of ones: ones_out2
    length_out2 = len(hex_array_out_tobytes_in_binary)
    # number of the whole string: ones_out2
    ones_and_length = torch.tensor([ones_out2, length_out2])
    return ones_and_length


def counter_ones_for_params_version_2(out):
    random_tensor_for_ones = torch.randn(len(out))
    random_tensor_for_length = torch.randn(len(out))
    for i in range(len(out)):
        random_tensor_for_ones[i] = counter_ones_version_2(out[i])[0]
        random_tensor_for_length[i] = counter_ones_version_2(out[i])[1]
    efficiency = torch.sum(random_tensor_for_ones)/torch.sum(random_tensor_for_length)
    return efficiency


def counter_ones_version_3(out):
    array_out= out.cpu().detach().numpy()
    # array_out= out.detach().numpy()
    array_out_tobytes= array_out.tobytes()
    hex_array_out_tobytes = bitstring.BitArray(array_out_tobytes)
    hex_array_out_tobytes_in_binary = hex_array_out_tobytes.bin
    ones_out2 = number_of_ones(hex_array_out_tobytes_in_binary)
    # # number of ones: ones_out2
    # length_out2 = len(hex_array_out_tobytes_in_binary)
    # # number of the whole string: ones_out2
    # ones_and_length = torch.tensor([ones_out2, length_out2])
    return ones_out2

def counter_ones_for_params_version_3(out):
    random_tensor_for_ones = torch.randn(len(out))
    for i in range(len(out)):
        random_tensor_for_ones[i] = counter_ones_version_2(out[i])[0]
        summation_of_ones = torch.sum(random_tensor_for_ones)
    return summation_of_ones

