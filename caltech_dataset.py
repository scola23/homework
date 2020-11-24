from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def _init_(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self)._init_(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        folder = list(os.path.split(root))[0]
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the _getitem_ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        self.labels = []
        self.data = []
        with open(folder + '/' + split + '.txt', 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                if line[:10].lower()!='background':
                    self.labels.append(line[:-15])
                    self.data.append(pil_loader(root + '/' + line))
                    
        labels_set = list(set(self.labels))
        
        for i in range(len(self.labels)):
            self.labels[i] = labels_set.index(self.labels[i])
                    
                
        

    def _getitem_(self, index):
        '''
        _getitem_ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        #image, label = ... # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int
        image = self.data[index]
        label = self.labels[index]
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return image, label

    def _len_(self):
        '''
        The _len_ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        #length = ... # Provide a way to get the length (number of elements) of the dataset
        return len(self.data)
