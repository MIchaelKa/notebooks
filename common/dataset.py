import os
import cv2

class FolderDataset():
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.image_names = sorted(os.listdir(root_path))

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.root_path, image_name)
        image = cv2.imread(image_path)
        return image