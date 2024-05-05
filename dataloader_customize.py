import pandas as pd
from torch.utils import data
import numpy as np

class EmoData(data.Dataset):
    def __init__(self, csv_file, transform):
        self.annotations = pd.read_csv(csv_file, header=0)
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        image = (np.array(self.annotations['image'][idx].split(' ')).astype('float32')/255.).reshape(48, 48)
        dist_emo = np.zeros(8, dtype="float32")
        dist_emo[self.annotations['emotion'][idx]] = 1.0
        if self.transform:
            image = self.transform(image)
        return {"image": image, "dist_emo": dist_emo}