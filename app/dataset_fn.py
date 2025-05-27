import torch, numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


train_transform = transforms.Compose([
    #transforms.Resize((178, 178)), # ---> VEDO SE METTENDO QUESTO E TOGLIENDO DAL DPWNOAD IL RESIZE L'IMMAGINE MGLIORA

    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.4588,0.4588,0.4588], std=[0.4588,0.4588,0.4588])
])

test_transform = transforms.Compose([
    #transforms.Resize((178, 178)),

    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.4588,0.4588,0.4588], std=[0.4588,0.4588,0.4588])
])

# VideoDataset object 

class VideoDataset(Dataset):
    def __init__(self, df, dataset_path, transform = None, t = 'single'):
        self.df = df
        self.transform = transform
        self.t = t
        self.dataset_path = dataset_path
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.transform is None: self.transform = transforms.ToTensor()
        images_path = self.df.iloc[index, 0]

        shots = self.df.iloc[index, -1] # Get the number of frames of a bag of shots
        # Each bag is half second frames

        images = None
        if self.t == 'single':
            # I want only the central frame
            images = self.transform(Image.open(f'{self.dataset_path}/{images_path}/shot{shots//2}.png'))

        elif self.t == 'early':
            # I want the 5 middle frames
            images = np.array([self.transform(Image.open(f'{self.dataset_path}/{images_path}/shot{idx}.png')).numpy() for idx in range((shots//2)-2,(shots//2)+3)])
        
        elif self.t == 'late':
            # I want the first and last frames
            images = np.array([self.transform(Image.open(f'{self.dataset_path}/{images_path}/shot0.png')).numpy(), self.transform(Image.open(f'{self.dataset_path}/{images_path}/shot{shots-1}.png')).numpy()])
        
        elif self.t == 'slow':
            # I want the 10 middle frames
            if shots%10 == 0:
                images = np.array([self.transform(Image.open(f'{self.dataset_path}/{images_path}/shot{idx}.png')).numpy() for idx in range((shots//2) - 5, (shots//2) + 5)])
            else:
                images = np.array([self.transform(Image.open(f'{self.dataset_path}/{images_path}/shot{idx}.png')).numpy() for idx in range((shots%10) - (shots%10)//2, shots-(shots%10)//2)])
        else:
            # Default to single frame if t is not recognized
            images = self.transform(Image.open(f'{self.dataset_path}/{images_path}/shot{shots//2}.png'))

        y_label = torch.tensor(np.where(self.df.iloc[index, 1:-1].to_numpy().astype(float) == 1.)[0][0]) # tensor with the corresponding label index (from 0 to 9)

        if self.t != 'single': images = torch.from_numpy(images) # Convert the numpy image to tensor in case I want the central frame

        return images, y_label, images_path
    


def spit_train(train_data, perc_val_size):
    '''
    PORPOUSE: 
      Split the dataset in train and validation set

    TAKES:
      - train_data: dataset to split
      - perc_val_size: percentage of split

    RETURNS: 
      Train dataset and validation dataset
    '''
    
    train_size = len(train_data)
    val_size = int((train_size * perc_val_size) // 100)
    train_size -= val_size

    return random_split(train_data, [int(train_size), int(val_size)]) #train_data, val_data 


def generate_dataloaders(train_data, val_data, test_data, batch_size, num_workers) -> tuple[DataLoader, DataLoader, DataLoader]:
    '''
    PORPOUSE:
      Generate the train vaiadtion and test dataloaders

    TAKES:
      - train_data: train dataset
      - val_data: validation dataset
      - test_data: test dataset
      - batch_size: specify how many bags of shots a bach must have
      - num_workers: specify how many workers will work on the dataloaders creation 

    RETURNS: 
      train, val and test dataloaders
    '''
    
    train_dl = DataLoader(dataset = train_data, batch_size = batch_size, num_workers = num_workers, shuffle = True)
    val_dl = DataLoader(dataset = val_data, batch_size = batch_size, num_workers = num_workers, shuffle = True)
    test_dl = DataLoader(dataset = test_data, batch_size = batch_size, num_workers = num_workers, shuffle = False)

    return train_dl, val_dl, test_dl




def get_torch_Dataloader(train_df, test_df, dataset_path, args):
    # Let's set the validations set to the 20% of the train dataset

    # Dataset for Single Frame
    train_data_single, val_data_single = spit_train(VideoDataset(df=train_df, dataset_path=dataset_path, transform=train_transform, t='single'), 20)
    test_data_single = VideoDataset(df=test_df, dataset_path=dataset_path, transform=test_transform, t='single')

    # Dataset for Multi Frame - Early Fusion
    train_data_early, val_data_early = spit_train(VideoDataset(df=train_df, dataset_path=dataset_path, transform=train_transform, t='early'), 20)
    test_data_early = VideoDataset(df=test_df, dataset_path=dataset_path, transform=test_transform, t='early')

    # Dataset for Multi Frame - Late Fusion
    train_data_late, val_data_late = spit_train(VideoDataset(df=train_df, dataset_path=dataset_path, transform=train_transform, t='late'), 20)
    test_data_late = VideoDataset(df=test_df, dataset_path=dataset_path, transform=test_transform, t='late')

    # Dataset for Multi Frame - Slow Fusion
    train_data_slow, val_data_slow = spit_train(VideoDataset(df=train_df, dataset_path=dataset_path, transform=train_transform, t='slow'), 20)
    test_data_slow = VideoDataset(df=test_df, dataset_path=dataset_path, transform=test_transform, t='slow')

    train_dl_single, val_dl_single, test_dl_single = generate_dataloaders(train_data_single, val_data_single, test_data_single, args.batch_size, args.num_workers)
    train_dl_late, val_dl_late, test_dl_late = generate_dataloaders(train_data_late, val_data_late, test_data_late, args.batch_size, args.num_workers)
    train_dl_early, val_dl_early, test_dl_early = generate_dataloaders(train_data_early, val_data_early, test_data_early, args.batch_size, args.num_workers)
    train_dl_slow, val_dl_slow, test_dl_slow = generate_dataloaders(train_data_slow, val_data_slow, test_data_slow, args.batch_size, args.num_workers)
    
    single = { 'train': train_dl_single, 'val': val_dl_single, 'test': test_dl_single }
    early = { 'train': train_dl_early, 'val': val_dl_early, 'test': test_dl_early }
    late = { 'train': train_dl_late, 'val': val_dl_late, 'test': test_dl_late }
    slow = { 'train': train_dl_slow, 'val': val_dl_slow, 'test': test_dl_slow}

    return {
        'SingleResFovea': single, 'SingleResContext': single, 'MultiRes': single, 
        'LateFusion': late, 'EarlyFusion': early, 'SlowFusion': slow
    }
