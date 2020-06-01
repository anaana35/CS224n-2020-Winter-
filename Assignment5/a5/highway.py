#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, e_word):
        super(Highway, self).__init__()
        self.e_word = e_word
        self.proj_layer = torch.nn.Linear(self.e_word, self.e_word)
        self.gate_layer = torch.nn.Linear(self.e_word, self.e_word)


    def forward(self, conv_out: torch.Tensor) -> torch.Tensor:
        # conv_out : batch_size,e_word
        # x_proj : batch_size,e_word
        # x_gate = batch_size,e_word
        x_proj = self.proj_layer(conv_out).relu()
        x_gate = self.gate_layer(conv_out).sigmoid()

        x_highway = x_gate * x_proj + (1 - x_gate) * conv_out
        return x_highway
    ### END YOUR CODE
