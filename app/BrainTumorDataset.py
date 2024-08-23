from monai.data import Dataset
class BrainTumorDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image and label
        image = self.image_paths[idx]

        # Apply the transform
        if self.transform:
            image = self.transform(image)

        return image