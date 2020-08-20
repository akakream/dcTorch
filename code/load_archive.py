import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataGenBigEarth import dataGenBigEarthLMDB, ToTensor, Normalize

def load_archive(bigEarthLMDBPth, train_csv_path, val_csv_path, test_csv_path, batch_size):

    bands_mean = {
                    'bands10_mean': [ 429.9430203 ,  614.21682446,  590.23569706, 2218.94553375],
                    'bands20_mean': [ 950.68368468, 1792.46290469, 2075.46795189, 2266.46036911, 1594.42694882, 1009.32729131],
                    'bands60_mean': [ 340.76769064, 2246.0605464 ],
                }

    bands_std = {
                    'bands10_std': [ 572.41639287,  582.87945694,  675.88746967, 1365.45589904],
                    'bands20_std': [ 729.89827633, 1096.01480586, 1273.45393088, 1356.13789355, 1079.19066363,  818.86747235],
                    'bands60_std': [ 554.81258967, 1302.3292881 ]
                }

    upsampling = True


    train_dataGen = dataGenBigEarthLMDB(
                    bigEarthPthLMDB=bigEarthLMDBPth,
                    state='train',
                    imgTransform=transforms.Compose([
                        ToTensor(),
                        Normalize(bands_mean, bands_std)
                    ]),
                    upsampling=upsampling,
                    train_csv=train_csv_path,
                    val_csv=val_csv_path,
                    test_csv=test_csv_path
    )

    val_dataGen = dataGenBigEarthLMDB(
                    bigEarthPthLMDB=bigEarthLMDBPth,
                    state='val',
                    imgTransform=transforms.Compose([
                        ToTensor(),
                        Normalize(bands_mean, bands_std)
                    ]),
                    upsampling=upsampling,
                    train_csv=train_csv_path,
                    val_csv=val_csv_path,
                    test_csv=test_csv_path
    )

    train_data_loader = DataLoader(train_dataGen, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(val_dataGen, batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True)

    return train_data_loader, val_data_loader
