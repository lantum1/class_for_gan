class FATTrainDataset(Dataset):
    def __init__(self, curated, noisy, transforms):
        super().__init__()
        self.curated = curated
        self.noisy = noisy
        self.transforms = transforms
        
    def __len__(self):
        return len(self.curated)
    
    def __getitem__(self, idx):
        # crop 1sec
        noisy_image = Image.fromarray(self.noisy[idx], mode='RGB')        
        time_dim, base_dim = noisy_image.size
        crop = random.randint(0, time_dim - base_dim)
        noisy_image = noisy_image.crop([crop, 0, crop + base_dim, base_dim])
        noisy_image = self.transforms(noisy_image).div_(255)
        
        curated_image = Image.fromarray(self.curated[idx], mode='RGB')        
        time_dim, base_dim = curated_image.size
        crop = random.randint(0, time_dim - base_dim)
        curated_image = curated_image.crop([crop, 0, crop + base_dim, base_dim])
        curated_image = self.transforms(curated_image).div_(255)
        
        return noisy_image, curated_image
