#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformer import layers


class PosSupervisor(nn.Module):

    def __init__(self, vocab_size, layer_dimension):
        super().__init__()
        self.vocab_size = vocab_size
        self.generator = layers.Generator(layer_dimension, vocab_size, rank=None)
        self.train_criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, src_enc, tgt):
        padded_tgt_loss = pad_sequence(tgt).transpose(0, 1)[:, 1:]
        out = self.generator(src_enc)
        norm = (padded_tgt_loss != 0).data.sum().item()
        loss = self.train_criterion(out.contiguous().view(-1, out.size(-1)),
                                    padded_tgt_loss.contiguous().view(-1)) / norm
        return loss
