import numpy as np
import os
import cv2
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL.Image as Image
import albumentations as albu
import albumentations.augmentations.functional as F
import PIL.Image as Image
import imgaug.augmenters as iaa

class BData(data.Dataset):
    '''
    The corresponding data set is loaded according to the value of phase.
    When the phase value is "train", the source domain dataset is loaded.
    When the phase value is "unlabel_data", the target domain unlabelled dataset is loaded.
    When the phase value is "reallabel_data", the target domain labelled dataset is loaded.
    When the phase value is "test", the target domain test dataset is loaded.
    '''

    def __init__(self, phase, args):
        super().__init__()
        self.phase = phase
        print("phase")
        print(str(phase)) 
        if phase == 'train':
            self.args = args
            labeled_files = args.source_txt.split(',')
            print(str(labeled_files))
            image_labels = []
            image_names = []
            for file_name in labeled_files:
                for line in open(file_name, 'r').readlines():
                    line = line.strip()
                    image_path = os.path.join(line.split(' ')[0])
                    label = int(line.split(' ')[-1])
                    image_labels.append(label)
                    image_names.append(image_path)
            self.image_labels = image_labels
            self.image_names = image_names

        elif phase == 'unlabel_data':
            self.args = args
            unlabeled_files = args.target_unlabeled_txt.split(',')
            print(str(unlabeled_files))
            print("\n")
            unlabeled_labels = []
            unlabeled_names = []
            id_code2index = {}
            test_id_codes = []
            for file_name in unlabeled_files:
                readlins = open(file_name, 'r').readlines()
                for i, line in enumerate(readlins):
                    line = line.strip()
                    image_path = os.path.join(line.split(' ')[0])
                    label = int(line.split(' ')[1])
                    realabel = int(line.split(' ')[2])
                    unlabeled_labels.append(label)
                    unlabeled_names.append(image_path)
                    id_code2index[line.split(' ')[0]] = i
                    test_id_codes.append(line.split(' ')[0])
            self.unlabeled_labels = unlabeled_labels
            self.unlabeled_names = unlabeled_names
            self.unlabeled_count = len(self.unlabeled_names)
            self.test_id_codes = test_id_codes
            self.id_code2index = id_code2index

        elif phase == 'reallabel_data':
            self.args = args
            reallabeled_files = args.target_labeled_txt.split(',')
            print(str(reallabeled_files))
            print("\n")
            reallabeled_labels = []
            reallabeled_names = []
            for file_name in reallabeled_files:
                for line in open(file_name, 'r').readlines():
                    line = line.strip()
                    # print('line',line)
                    image_path = os.path.join(line.split(' ')[0])
                    label = int(line.split(' ')[-1])
                    reallabeled_labels.append(label)
                    reallabeled_names.append(image_path)
            self.reallabeled_labels = reallabeled_labels
            self.reallabeled_names = reallabeled_names

        else:
            unlabeled_labels = []
            unimage_names = []
            unlabeled_files = args.target_test_txt.split(',')
            print(str(unlabeled_files))
            print("\n")
            for file_name in unlabeled_files:
                readlins = open(file_name, 'r').readlines()
                for i, line in enumerate(readlins):
                    line = line.strip()
                    image_path = os.path.join(line.split(' ')[0])
                    label = int(line.split(' ')[-1])
                    unlabeled_labels.append(label)
                    unimage_names.append(image_path)
            self.test_labels = unlabeled_labels
            self.unimage_names = unimage_names
            self.unlabeled_count = len(self.unimage_names)

    def __getitem__(self, index):
        basedir = args.txt_dir
        if self.phase == 'train':
            kind = 5
            labeled_image = cv2.imdecode(np.fromfile(os.path.join(basedir, self.image_names[index]), dtype=np.uint8),
                                         -1)
            labeled_image = cv2.resize(labeled_image, (512, 255))
            labeled_image = {'image': labeled_image, }
            labeled_image = labeled_image['image']
            mode = random.randint(0, 2 * kind - 1)
            labeled_image = np.transpose(labeled_image, (2, 0, 1))
            labeled_image = torch.from_numpy(labeled_image).float().div(255)
            labeled_label = self.image_labels[index]
            return labeled_image, labeled_label

        elif self.phase == 'unlabel_data':
            kind = 5
            unlabeled_image = cv2.imdecode(
                np.fromfile(os.path.join(basedir, self.unlabeled_names[index % self.unlabeled_count]), dtype=np.uint8),
                -1)
            unlabeled_image = cv2.resize(unlabeled_image, (512, 255))
            unlabeled_image = {'image': unlabeled_image, }
            unlabeled_image = unlabeled_image['image']
            unlabeled_image = np.transpose(unlabeled_image, (2, 0, 1))
            unlabeled_image = torch.from_numpy(unlabeled_image).float().div(255)
            unlabeled_label = self.unlabeled_labels[index % self.unlabeled_count]
            unlabeled_id = index % self.unlabeled_count
            return unlabeled_image, unlabeled_label, self.test_id_codes[unlabeled_id]


        elif self.phase == 'reallabel_data':
            kind = 5
            reallabeled_image = cv2.imdecode(
                np.fromfile(os.path.join(basedir, self.reallabeled_names[index]), dtype=np.uint8), -1)
            reallabeled_image = cv2.resize(reallabeled_image, (512, 255))
            reallabeled_image = {'image': reallabeled_image, }
            reallabeled_image = reallabeled_image['image']
            mode = random.randint(0, 2 * kind - 1)
            reallabeled_image = np.transpose(reallabeled_image, (2, 0, 1))
            reallabeled_image = torch.from_numpy(reallabeled_image).float().div(255)
            reallabeled_label = self.reallabeled_labels[index]
            return reallabeled_image, reallabeled_label

        else:
            unlabeled_image = cv2.imdecode(
                np.fromfile(os.path.join(basedir, self.unimage_names[index]), dtype=np.uint8), -1)
            unlabeled_image = cv2.resize(unlabeled_image, (512, 255))
            unlabeled_image = {'image': unlabeled_image, }
            unlabeled_image = unlabeled_image['image']
            unlabeled_image = np.transpose(unlabeled_image, (2, 0, 1))
            unlabeled_image = torch.from_numpy(unlabeled_image).float().div(255)
            return unlabeled_image, self.test_labels[index]

    def __len__(self):
        if self.phase == 'train':
            return len(self.image_names)
        elif self.phase == 'unlabel_data':
            return len(self.unlabeled_names)
        elif self.phase == 'reallabel_data':
            return len(self.reallabeled_names)
        else:
            return len(self.unimage_names)

    def update(self, id_code2label):

        '''
        Update unlabeled_labels
        '''
        for id_code in id_code2label:
            self.unlabeled_labels[
                self.id_code2index[id_code]
            ] = id_code2label[id_code]

