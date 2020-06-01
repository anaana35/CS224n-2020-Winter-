#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    pass

    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, e_char, kernel_size, e_word):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.e_word = e_word
        self.e_char = e_char
        self.conv = nn.Conv1d(in_channels=self.e_char, out_channels=self.e_word,
                              kernel_size=self.kernel_size,
                              padding=1)

    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        # sentence_length, batch_size, e_char, max_word_length
        # 输入维度：batch_size,e_char,m_word
        # 输出维度：batch_size, e_word
        x_conv = self.conv(x_reshaped)
        x_conv_out,_ = torch.max(x_conv.relu(),dim=-1)
        return x_conv_out

    ### END YOUR CODE
