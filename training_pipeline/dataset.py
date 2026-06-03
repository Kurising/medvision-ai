import os
from torch.utils.data import Dataset
from PIL import Image

class BrainDataset(Dataset):

    def __init__(self , root_dir , transform=None):
        # runs ONCE when you create the dataset object
        # collect all image paths + labels into a list
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
        self.samples = []
        
        for folder_name in os.listdir(root_dir):
            label = self.classes[folder_name]
            folder_path = os.path.join(root_dir, folder_name)
            
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                self.samples.append((image_path, label))

    def __len__(self):
        # returns total number of images
        return len(self.samples)

    def __getitem__(self,idx):
        # loads ONE image by index
        # returns (image_tensor, label)
        image_path, label = self.samples[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label
