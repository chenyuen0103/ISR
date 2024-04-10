import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from data.confounder_dataset import ConfounderDataset
from configs import model_attributes
from PIL import Image
from torch.utils.data import Subset



class PACSDataset(ConfounderDataset):
    def __init__(self, root_dir, target_name, confounder_names,
                 model_type, augment_data, train_transform=None, eval_transform=None, ):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data
        self.data = []
        self.confounder_array = []
        self.filename_array = []
        self.y_array = []
        self.group_array = []
        self.split_array = []
        self.n_classes = 7
        self.n_groups = pow(7, 4)

        # Initialize transforms
        # self.train_transform = None
        # self.eval_transform = None
        target_resolution = model_attributes[model_type]['target_resolution']

        self.transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]) ,int(target_resolution[0]))),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        self._prepare_dataset()
        # breakpoint()
        self.group_array = (self.y_array * (self.n_groups / self.n_classes) + self.confounder_array).astype('int')

    def _prepare_dataset(self):
        self.label_map = {'dog': 0, 'elephant': 1, 'giraffe': 2, 'guitar': 3, 'horse': 4, 'house': 5, 'person': 6}
        self.environments = ['art_painting', 'cartoon', 'photo', 'sketch']
        self.env_map = {env: idx for idx, env in enumerate(self.environments)}

        self.test_env = self.environments[-1]
        self.train_envs = [env for env in self.environments if env != self.test_env]

        for env in self.train_envs:
            domain_path = os.path.join(self.root_dir, env)
            for label in os.listdir(domain_path):
                label_path = os.path.join(domain_path, label)
                for img_filename in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_filename)
                    self.data.append(img_path)
                    self.filename_array.append(img_filename)
                    self.y_array.append(self.label_map[label])
                    self.confounder_array.append(self.env_map[env])

        # split the training data into train and val, and exclude the test environment
        self.y_array = np.array(self.y_array)
        self.confounder_array = np.array(self.confounder_array)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        y = self.y_array[idx]
        # breakpoint()
        domain = self.confounder_array[idx]
        g = self.group_array[idx]
        img = Image.open(img_path).convert('RGB')

        # Apply the appropriate transform
        if self.augment_data and domain != self.env_map[self.test_env]:
            img = self.augment_transform(img)
        else:
            img = self.transform(img)

        # Flatten if needed
        if model_attributes[self.model_type]['flatten']:
            assert img.dim() == 3
            img = img.view(-1)
        x = img

        return x, y, g


    def get_splits(self, splits, train_frac=1.0, val_frac=0.2):
        subsets = {}
        train_val_domains = [env for env in self.train_envs]
        train_val_indices = [i for i, domain in enumerate(self.train_envs) if domain in train_val_domains]

        # Shuffle train_val_indices to randomize the train/val split
        np.random.shuffle(train_val_indices)

        num_train_val = len(train_val_indices)
        num_val = int(np.round(num_train_val * val_frac))
        num_train = num_train_val - num_val

        # Ensure we respect the train_frac by adjusting the number of training samples
        num_train = int(np.round(num_train * train_frac))

        train_indices = train_val_indices[:num_train]
        val_indices = train_val_indices[num_train:num_train + num_val]

        # Test indices are simply all indices from the test environment
        test_indices = [i for i, domain in enumerate(self.test_env)]

        if 'train' in splits:
            subsets['train'] = Subset(self, train_indices)
        if 'val' in splits:
            subsets['val'] = Subset(self, val_indices)
        if 'test' in splits:
            subsets['test'] = Subset(self, test_indices)

        return subsets

    def group_str(self, group_idx):
        """
        Generates a string representing the group based on the provided index.
        This version assumes group_idx is directly mapping to domain and label indices.
        """
        # Assuming there's a way to calculate domain_idx and label_idx from group_idx
        domain_idx = self._calculate_domain_idx(group_idx)  # Needs implementation
        label_idx = self._calculate_label_idx(group_idx)  # Needs implementation

        domain_name = self.environments[domain_idx]  # List of domains
        label_name = self.labels[label_idx]  # List of labels

        group_name = f"Domain = {domain_name}, Label = {label_name}"
        return group_name