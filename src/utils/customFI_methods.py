from pytorchfi.neuron_error_models import (
    random_batch_element,
    random_neuron_location,
)
from pytorchfi import core
import random
import logging
import torch
import struct
from random import randint
from src.utils.helpers import device


"""
This is our custom fi function to target exact neuron 
"""


def random_neuron_single_bit_inj_Aman(
    pfi: core.fault_injection, layer_ranges, layer, fi_c, fi_h, fi_w, bit_pos
):
    # TODO Support multiple error models via list
    pfi.set_conv_max(layer_ranges)

    batch = random_batch_element(pfi)
    (returned_layer, C, H, W) = random_neuron_location(pfi, layer)
    if returned_layer != layer:
        print("Problem: fi_layer is not equal to the returned layer")
    return pfi.declare_neuron_fi(
        batch=[batch],
        layer_num=[layer],
        # dim1=[C],
        # dim2=[H],
        # dim3=[W],
        dim1=[fi_c],
        dim2=[fi_h],
        dim3=[fi_w],
        function=pfi.single_bit_flip_signed_across_batch_Aman,
    )


class single_bit_flip_func(core.fault_injection):
    def __init__(self, model, batch_size, input_shape=None, bit_pos=None, **kwargs):
        if input_shape is None:
            input_shape = [3, 224, 224]

        super().__init__(model, batch_size, input_shape=input_shape, bit_pos=bit_pos, **kwargs)
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")
        self.bit_pos = bit_pos
        if bit_pos is None or -1:
            bit_pos = randint(0, 31)
        self.bits = kwargs.get("bits", 8)
        self.LayerRanges = []

    def set_conv_max(self, data):
        self.LayerRanges = data

    def reset_conv_max(self, data):
        self.LayerRanges = []

    def get_conv_max(self, layer):
        return self.LayerRanges[layer]

    @staticmethod
    def _twos_comp(val, bits):
        if (val & (1 << (bits - 1))) != 0:
            val = val - (1 << bits)
        return val

    def _twos_comp_shifted(self, val, nbits):
        return (1 << nbits) + val if val < 0 else self._twos_comp(val, nbits)

    """
    custom bitflip function
    """

    def _flip_bit_signed_Aman(self, orig_value, max_value, bit_pos):

        save_type = orig_value.dtype
        packed = struct.pack("!f", orig_value)
        integer_representation = int.from_bytes(packed, byteorder="big")
        binary_representation = bin(integer_representation)[2:].zfill(32)
        logging.info(f"Original Value: {orig_value}")
        logging.info(f"Orginal bits: {binary_representation}")
        bit_pos = self.bit_pos

        if bit_pos == -1:
            return torch.tensor(orig_value, dtype=save_type, device=device)

        bit_list = list(binary_representation)
        if bit_pos >= len(bit_list):
            logging.info(f"bit_pos > len(binary_representation)")
            bit_pos = 1
        bit_list[bit_pos] = "1" if bit_list[bit_pos] == "0" else "0"
        new_binary_representation = "".join(bit_list)
        new_packed = int(new_binary_representation, 2).to_bytes(4, byteorder="big")
        new_number = struct.unpack("!f", new_packed)[0]
        logging.info(f"New bits    : {new_binary_representation}")
        logging.info(f"New Number: {new_number}")
        return torch.tensor(new_number, dtype=save_type, device=device)

    def _flip_bit_signed(self, orig_value, max_value, bit_pos):
        # quantum value
        save_type = orig_value.dtype
        total_bits = self.bits
        logging.info("Original Value: %d", orig_value)

        quantum = int((orig_value / max_value) * (2.0 ** (total_bits - 1)))
        twos_comple = self._twos_comp_shifted(quantum, total_bits)  # signed
        logging.info("Quantum: %d", quantum)
        logging.info("Twos Couple: %d", twos_comple)

        # binary representation
        bits = bin(twos_comple)[2:]
        logging.info("Bits: %s", bits)

        # sign extend 0's
        temp = "0" * (total_bits - len(bits))
        bits = temp + bits
        if len(bits) != total_bits:
            raise AssertionError
        logging.info("sign extend bits %s", bits)

        # flip a bit
        # use MSB -> LSB indexing
        if bit_pos >= total_bits:
            raise AssertionError

        bits_new = list(bits)
        bit_loc = total_bits - bit_pos - 1
        if bits_new[bit_loc] == "0":
            bits_new[bit_loc] = "1"
        else:
            bits_new[bit_loc] = "0"
        bits_str_new = "".join(bits_new)
        logging.info("New bits: %s", bits_str_new)

        # GPU contention causes a weird bug...
        if not bits_str_new.isdigit():
            logging.info("Error: Not all the bits are digits (0/1)")

        # convert to quantum
        if not bits_str_new.isdigit():
            raise AssertionError
        new_quantum = int(bits_str_new, 2)
        out = self._twos_comp(new_quantum, total_bits)
        logging.info("Out: %s", out)

        # get FP equivalent from quantum
        new_value = out * ((2.0 ** (-1 * (total_bits - 1))) * max_value)
        logging.info("New Value: %d", new_value)

        return torch.tensor(new_value, dtype=save_type)

    """
    this single_bit_flip_signed_across_batch_Aman is our own custom bitflip function
    """

    def single_bit_flip_signed_across_batch_Aman(self, module, input_val, output):
        corrupt_conv_set = self.get_corrupt_layer()
        range_max = self.get_conv_max(self.get_current_layer())
        logging.info("Current layer: %s", self.get_current_layer())
        logging.info("Range_max: %s", range_max)

        if type(corrupt_conv_set) is list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.get_current_layer(),
                    range(len(corrupt_conv_set)),
                )
            )
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                prev_value = output[self.corrupt_batch[i]][self.corrupt_dim1[i]][
                    self.corrupt_dim2[i]
                ][self.corrupt_dim3[i]]

                bit_pos = self.bit_pos
                logging.info("Random Bit: %d", bit_pos)
                new_value = self._flip_bit_signed_Aman(prev_value, range_max, bit_pos)

                output[self.corrupt_batch[i]][self.corrupt_dim1[i]][self.corrupt_dim2[i]][
                    self.corrupt_dim3[i]
                ] = new_value

        else:
            if self.get_current_layer() == corrupt_conv_set:
                prev_value = output[self.corrupt_batch][self.corrupt_dim1][self.corrupt_dim2][
                    self.corrupt_dim3
                ]

                bit_pos = self.bit_pos
                logging.info("Random Bit: %d", rand_bit)
                new_value = self._flip_bit_signed_Aman(prev_value, range_max, bit_pos)

                output[self.corrupt_batch][self.corrupt_dim1][self.corrupt_dim2][
                    self.corrupt_dim3
                ] = new_value

        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()

    # def single_bit_flip_signed_across_batch_Aman(self, module, input_val, output):
    #         corrupt_conv_set = self.corrupt_layer
    #         range_max = self.get_conv_max(self.current_layer)
    #         logging.info(f"Current layer: {self.current_layer}")
    #         logging.info(f"Range_max: {range_max}")

    #         if type(corrupt_conv_set) is list:
    #             inj_list = list(
    #                 filter(
    #                     lambda x: corrupt_conv_set[x] == self.current_layer,
    #                     range(len(corrupt_conv_set)),
    #                 )
    #             )
    #             for i in inj_list:
    #                 self.assert_injection_bounds(index=i)
    #                 prev_value = output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
    #                     self.corrupt_dim[1][i]
    #                 ][self.corrupt_dim[2][i]]

    #                 #rand_bit = random.randint(0, self.bits - 1)
    #                 logging.info(f"Random Bit: {rand_bit}")
    #                 new_value = self._flip_bit_signed_Aman(prev_value, range_max)

    #                 output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
    #                     self.corrupt_dim[1][i]
    #                 ][self.corrupt_dim[2][i]] = new_value

    #         else:
    #             if self.current_layer == corrupt_conv_set:
    #                 prev_value = output[self.corrupt_batch][self.corrupt_dim[0]][
    #                     self.corrupt_dim[1]
    #                 ][self.corrupt_dim[2]]

    #                 rand_bit = random.randint(0, self.bits - 1)
    #                 logging.info(f"Random Bit: {rand_bit}")
    #                 new_value = self._flip_bit_signed_Aman(prev_value, range_max)

    #                 output[self.corrupt_batch][self.corrupt_dim[0]][self.corrupt_dim[1]][
    #                     self.corrupt_dim[2]
    #                 ] = new_value

    #         self.updateLayer()
    #         if self.current_layer >= len(self.output_size):
    #             self.reset_current_layer()
