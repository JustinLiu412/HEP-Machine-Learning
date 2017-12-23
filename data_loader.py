from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pdb
from adlkit.data_provider.data_providers import H5FileDataProvider

class LCDDataset(Dataset): 
    """Dataset"""

    def __init__(self, listOfFilePath, batch_size, label_map, normalize=True, phase='Train'): 
        """
        '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_0.h5'
        '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_0.h5'
        """
        # pdb.set_trace()
        self.label_map = label_map
        self.normalize = normalize
        self.batch_size = batch_size
        if phase == 'Train': 
            self.specification = [
                             [listOfFilePath[0], ['ECAL', 'HCAL'], 'class_0', 1],
                             [listOfFilePath[1], ['ECAL', 'HCAL'], 'class_1', 2], 
                             [listOfFilePath[2], ['ECAL', 'HCAL'], 'class_0', 3],
                             [listOfFilePath[3], ['ECAL', 'HCAL'], 'class_1', 4],
                             [listOfFilePath[4], ['ECAL', 'HCAL'], 'class_0', 5],
                             [listOfFilePath[5], ['ECAL', 'HCAL'], 'class_1', 6],
                             [listOfFilePath[6], ['ECAL', 'HCAL'], 'class_0', 7],
                             [listOfFilePath[7], ['ECAL', 'HCAL'], 'class_1', 8],
                             [listOfFilePath[8], ['ECAL', 'HCAL'], 'class_0', 9],
                             [listOfFilePath[9], ['ECAL', 'HCAL'], 'class_1', 10],
                             [listOfFilePath[10], ['ECAL', 'HCAL'], 'class_0', 11],
                             [listOfFilePath[11], ['ECAL', 'HCAL'], 'class_1', 12],
                             [listOfFilePath[12], ['ECAL', 'HCAL'], 'class_0', 13],
                             [listOfFilePath[13], ['ECAL', 'HCAL'], 'class_1', 14],
                             [listOfFilePath[14], ['ECAL', 'HCAL'], 'class_0', 15],
                             [listOfFilePath[15], ['ECAL', 'HCAL'], 'class_1', 16],
                             [listOfFilePath[16], ['ECAL', 'HCAL'], 'class_0', 17],
                             [listOfFilePath[17], ['ECAL', 'HCAL'], 'class_1', 18],
                             [listOfFilePath[18], ['ECAL', 'HCAL'], 'class_0', 19],
                             [listOfFilePath[19], ['ECAL', 'HCAL'], 'class_1', 20], 
                             [listOfFilePath[20], ['ECAL', 'HCAL'], 'class_0', 21],
                             [listOfFilePath[21], ['ECAL', 'HCAL'], 'class_1', 22], 
                             [listOfFilePath[22], ['ECAL', 'HCAL'], 'class_0', 23],
                             [listOfFilePath[23], ['ECAL', 'HCAL'], 'class_1', 24], 
                             [listOfFilePath[24], ['ECAL', 'HCAL'], 'class_0', 25],
                             [listOfFilePath[25], ['ECAL', 'HCAL'], 'class_1', 26], 
                             [listOfFilePath[26], ['ECAL', 'HCAL'], 'class_0', 27],
                             [listOfFilePath[27], ['ECAL', 'HCAL'], 'class_1', 28], 
                             [listOfFilePath[28], ['ECAL', 'HCAL'], 'class_0', 29],
                             [listOfFilePath[29], ['ECAL', 'HCAL'], 'class_1', 30], 
                             [listOfFilePath[30], ['ECAL', 'HCAL'], 'class_0', 31],
                             [listOfFilePath[31], ['ECAL', 'HCAL'], 'class_1', 32], 
                             [listOfFilePath[32], ['ECAL', 'HCAL'], 'class_0', 33],
                             [listOfFilePath[33], ['ECAL', 'HCAL'], 'class_1', 34], 
                             [listOfFilePath[34], ['ECAL', 'HCAL'], 'class_0', 35],
                             [listOfFilePath[35], ['ECAL', 'HCAL'], 'class_1', 36], 
                             [listOfFilePath[36], ['ECAL', 'HCAL'], 'class_0', 37],
                             [listOfFilePath[37], ['ECAL', 'HCAL'], 'class_1', 38], 
                             [listOfFilePath[38], ['ECAL', 'HCAL'], 'class_0', 39],
                             [listOfFilePath[39], ['ECAL', 'HCAL'], 'class_1', 40]
                            ]
        elif phase == 'Test': 
            self.specification = [
                             [listOfFilePath[0], ['ECAL', 'HCAL'], 'class_0', 1],
                             [listOfFilePath[1], ['ECAL', 'HCAL'], 'class_1', 2], 
                             [listOfFilePath[2], ['ECAL', 'HCAL'], 'class_0', 3],
                             [listOfFilePath[3], ['ECAL', 'HCAL'], 'class_1', 4],
                             [listOfFilePath[4], ['ECAL', 'HCAL'], 'class_0', 5],
                             [listOfFilePath[5], ['ECAL', 'HCAL'], 'class_1', 6],
                             [listOfFilePath[6], ['ECAL', 'HCAL'], 'class_0', 7],
                             [listOfFilePath[7], ['ECAL', 'HCAL'], 'class_1', 8],
                             [listOfFilePath[8], ['ECAL', 'HCAL'], 'class_0', 9],
                             [listOfFilePath[9], ['ECAL', 'HCAL'], 'class_1', 10],
                             [listOfFilePath[10], ['ECAL', 'HCAL'], 'class_0', 11],
                             [listOfFilePath[11], ['ECAL', 'HCAL'], 'class_1', 12],
                             [listOfFilePath[12], ['ECAL', 'HCAL'], 'class_0', 13],
                             [listOfFilePath[13], ['ECAL', 'HCAL'], 'class_1', 14],
                             [listOfFilePath[14], ['ECAL', 'HCAL'], 'class_0', 15],
                             [listOfFilePath[15], ['ECAL', 'HCAL'], 'class_1', 16],
                             [listOfFilePath[16], ['ECAL', 'HCAL'], 'class_0', 17],
                             [listOfFilePath[17], ['ECAL', 'HCAL'], 'class_1', 18],
                             [listOfFilePath[18], ['ECAL', 'HCAL'], 'class_0', 19],
                             [listOfFilePath[19], ['ECAL', 'HCAL'], 'class_1', 20]
                            ]
        self.tmp_data_provider = H5FileDataProvider(self.specification, 
                               batch_size=self.batch_size, 
                               n_readers=2,
                                   q_multipler=3,
                                               wrap_examples=True, 
                               use_shared_memory=False, 
                               read_multiplier=1, 
                               make_file_index=True)
        self.tmp_data_provider.start()

        
    def __len__(self):
        return len(self.specification)*10000

    def getbatch(self):
        self.thing = self.tmp_data_provider.first().generate().next()
        image = torch.from_numpy(np.expand_dims(self.thing[0], axis=1))
        if self.normalize: 
            image = image.div(image.max() - image.min())
        # pdb.set_trace()
        label = [self.label_map[x[0].split('/')[-1].split('.')[0].split('_')[0]] for x in self.thing[-1]]
        label = torch.LongTensor(label)
        # label = self.thing[-1][0].split('/')[-1].split('.')[0].split('_')[0]
        
        sample = {'data': image, 'label': label}

        return image, label

    def __getitem__(self, idx):
        self.thing = self.tmp_data_provider.first().generate().next()
        image = torch.from_numpy(np.expand_dims(self.thing[0][idx], axis=1))
        if self.normalize: 
            image = image.div(image.max() - image.min())
        # pdb.set_trace()
        # label = [self.label_map[x[0].split('/')[-1].split('.')[0].split('_')[0]] for x in self.thing[-1]]
        label = self.thing[-1][idx][0].split('/')[-1].split('.')[0].split('_')[0]
        pdb.set_trace
        label = torch.LongTensor(label)
        
        sample = {'data': image, 'label': label}

        return image, label



class BatchLCDDataset(Dataset): 
    """
    """
    def __init__(self, listOfFilePath, batch_size, normalize=True): 
        """
        '/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_0.h5'
        '/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_0.h5'
        """
        self.normalize = normalize
        self.batch_size = batch_size
        self.specification = [
                         [listOfFilePath[0], ['ECAL', 'HCAL'], 'class_1', 1],
                         [listOfFilePath[1], ['ECAL', 'HCAL'], 'class_2', 2],
                         [listOfFilePath[2], ['ECAL', 'HCAL'], 'class_3', 3],
                         [listOfFilePath[3], ['ECAL', 'HCAL'], 'class_4', 4],
                        ]
        self.tmp_data_provider = H5FileDataProvider(self.specification, 
                               batch_size=self.batch_size, 
                               n_readers=2,
                                   q_multipler=3,
                                               wrap_examples=True, 
                               use_shared_memory=False, 
                               read_multiplier=1, 
                               make_file_index=True)
        self.tmp_data_provider.start()

    def __len__(self):
        return len(self.specification)*10000






