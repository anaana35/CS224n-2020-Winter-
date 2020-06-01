#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()
        self.word_embed_size = word_embed_size
        self.vocab = vocab
        m_word = len(self.vocab.char2id)
        pad_token_idx = self.vocab.char2id['<pad>']
        self.embedding = nn.Embedding(num_embeddings=m_word, embedding_dim=50, padding_idx=pad_token_idx)
        self.conv = CNN(e_char=50, kernel_size=5, e_word=self.word_embed_size)
        self.highway = Highway(self.word_embed_size)
        self.dropout = nn.Dropout(0.3)

        ### YOUR CODE HERE for part 1h

        ### END YOUR CODE

    def forward(self, input):
        # xpadded to xword emb
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        (sentence_length, batch_size, max_word_length) = input.shape
        x_reshaped = self.embedding(input).contiguous().view(sentence_length * batch_size, -1, max_word_length)
        x_conv = self.conv(x_reshaped)
        x_highway = self.highway(x_conv)
        x_word_emb = self.dropout(x_highway).contiguous().view(sentence_length, batch_size, -1)
        return x_word_emb

        ### END YOUR CODE
