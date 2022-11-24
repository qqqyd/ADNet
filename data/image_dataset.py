import functools
import logging
import bisect

import torch.utils.data as data
import cv2
import numpy as np
import glob
from concern.config import Configurable, State
import math
import os
from PIL import Image

class ImageDataset(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    data_dir = State()
    txt_dir = State()
    data_list = State()
    processes = State(default=[])

    def __init__(self, data_dir=None, txt_dir=None, data_list=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        self.txt_dir = txt_dir or self.txt_dir
        self.data_list = data_list or self.data_list
        if 'train' in self.data_list[0]:
            self.is_training = True
        else:
            self.is_training = False
        self.debug = cmd.get('debug', False)
        self.image_paths = []
        self.gt_paths = []
        self.get_all_samples()
        self.change_dilate = False

    def get_all_samples(self):
        for i in range(len(self.data_dir)):
            with open(self.data_list[i], 'r') as fid:
                image_list = fid.readlines()
            if self.is_training:
                if 'SynthText' in self.data_list[i]:
                    image_path = []
                    gt_path = []
                    for timg in image_list:
                        gt_path.append(self.txt_dir[i] + timg[:-5] + '.txt')
                        temp = timg.strip().split('-')
                        image_path.append(self.data_dir[i] + os.path.join(temp[0], temp[1]))
                elif 'ctw' in self.data_list[i]:
                    image_path = [self.data_dir[i] + timg.strip() for timg in image_list]
                    gt_path = [self.txt_dir[i] + timg.strip()[:-4] + '.txt' for timg in image_list]
                else:
                    image_path = [self.data_dir[i]+timg.strip() for timg in image_list]
                    gt_path = [self.txt_dir[i]+timg.strip()+'.txt' for timg in image_list]
            else:
                image_path = [self.data_dir[i] + timg.strip() for timg in image_list]
                if 'TD500' in self.data_list[i] or 'total_text' in self.data_list[i]:
                    gt_path = [self.txt_dir[i] + timg.strip()+'.txt' for timg in image_list]
                elif 'ctw' in self.data_list[i]:
                    gt_path = [self.txt_dir[i] + timg.strip()[:-4] + '.txt' for timg in image_list]
                else:
                    gt_path = [self.txt_dir[i]+'gt_'+timg.strip().split('.')[0]+'.txt' for timg in image_list]
            self.image_paths += image_path
            self.gt_paths += gt_path
        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)

    def load_ann(self):
        res = []
        for gt in self.gt_paths:
            lines = []
            reader = open(gt, 'r', encoding='utf-8').readlines()
            for line in reader:
                item = {}
                parts = line.strip().split(',')
                label = parts[-1]
                if label == '1':
                    label = '###'

                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                if 'icdar' in self.data_dir[0]:
                    poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                else:
                    num_points = math.floor((len(line) - 1) / 2) * 2
                    poly = np.array(list(map(float, line[:num_points]))).reshape((-1, 2)).tolist()
                item['poly'] = poly
                item['text'] = label
                lines.append(item)
            res.append(lines)
        return res

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.image_paths[index]

        tmp = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if tmp is None:
            tmp = Image.open(image_path).convert('RGB')
        img = np.array(tmp).astype('float32')
        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        data['image'] = img
        target = self.targets[index]
        data['lines'] = target
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)
        return data

    def __len__(self):
        return len(self.image_paths)
