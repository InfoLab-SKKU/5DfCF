import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class ADNIDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with columns: 'subject_id','image_path','diagnosis',...
            root_dir (string): Directory with all the MRI images.
            transform (callable, optional): Optional transform applied on an image.
            target_transform (callable, optional): Optional transform applied on the target.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        # Encode diagnosis labels (e.g., CN, MCI, AD)
        self.label_encoder = LabelEncoder()
        self.annotations['diagnosis_label'] = self.label_encoder.fit_transform(
            self.annotations['diagnosis'].values
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        record = self.annotations.iloc[idx]
        img_file = record['image_path']
        img_path = os.path.join(self.root_dir, img_file)

        # Load NIfTI file
        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata()

        # Intensity normalization to [0,1]
        img_data = (img_data - np.min(img_data)) / (np.ptp(img_data) + 1e-8)

        # Convert to torch tensor and add channel dimension
        img_tensor = torch.from_numpy(img_data).unsqueeze(0).float()

        label = int(record['diagnosis_label'])
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Apply optional transforms
        if self.transform:
            img_tensor = self.transform(img_tensor)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)

        return {
            'image': img_tensor,
            'label': label_tensor,
            'subject_id': record['subject_id']
        }

# Example usage:
# dataset = ADNIDataset(
#     csv_file='adni_labels.csv',
#     root_dir='/path/to/adni/images/',
#     transform=None
# )
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
