import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    
############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################

## ['COAVAREP', 'FACET 4.2', 'OpenFace_2.0', 'All Labels', 'glove_vectors', 'OpenSMILE']
class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False, norm_lables=False):
        super(Multimodal_Datasets, self).__init__()
        _dataset_path = dataset_path
        dataset_path = os.path.join(dataset_path, data+'_data.data')
        dataset = torch.load(dataset_path)

        _vision = 'FACET 4.2'
        _audio = 'COAVAREP' 
        _text = 'glove_vectors' 
        _labels = 'All Labels'
        _vision_2 = 'OpenFace_2.0'
        _audio_2 = 'OpenSMILE'
        # These are torch tensors
        ## vision 
        self.vision = dataset[split_type][_vision].clone().cpu().detach().float()

        self.vision_2 = dataset[split_type][_vision_2].clone().cpu().detach().float()

        self.text = dataset[split_type][_text].clone().cpu().detach().float()
        ## audio
        self.audio = dataset[split_type][_audio]
        self.audio[self.audio == -float('inf')] = 0
        self.audio = self.audio.clone().cpu().detach().float()

        self.audio_2 = dataset[split_type][_audio_2]
        self.audio_2[self.audio_2 == -float('inf')] = 0
        self.audio_2 = self.audio_2.clone().cpu().detach().float()
        ## label
        self.labels = dataset[split_type][_labels][:, :, :1].clone().squeeze(1).cpu().detach().float()

        if norm_lables:
            self.labels = self.normalize(self.labels)
        assert self.labels.size(1) == 1
        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
         
        self.data = data
        
        self.n_modalities = 3 # vision/ text/ audio
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.audio_2.shape[1], self.vision.shape[1], self.vision_2.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.audio_2.shape[2], self.vision.shape[2], self.vision_2.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.audio_2[index], self.vision[index],  self.vision_2[index])
        Y = self.labels[index]
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)
        return X, Y, META        
    def normalize(self, x): # input shape 50 x features
        _eps = 1e-8
        print(f'X shape is : {x.shape}')
        for index in range(x.shape[0]):
            for seq in range(x[index].shape[0]):
                x[index][seq] = (x[index][seq] - x[index][seq].min())/(x[index][seq].max()- x[index][seq].min() + _eps)
                # x[index][seq] = (x[index][seq] - 1/2) * 2
                assert torch.isnan(x[index][seq]).sum().item() == 0
        return x