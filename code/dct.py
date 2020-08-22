import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from model import Net
from run_together import run_together
from load_archive import load_archive
import argparse

def add_arguments():
    ap = argparse.ArgumentParser(prog='discrepant collaborative training for multi label learning', 
            description='This is the Bachelor"s Thesis of Ahmet Kerem Aksoy at Technical University of Berlin. Date: May 7, 2020.', 
            epilog='-- The computer was born to solve problems that did not exist before. --')
    ap.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size, default is 32')
    ap.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs, default is 10')
    ap.add_argument('-lp', '--lmdb_path', help='Give the path to the folder where the lmdb file is located.')
    ap.add_argument('-tr', '--train_csv_path', help='Give the path to the train.csv file.')
    ap.add_argument('-vl', '--val_csv_path', help='Give the path to the val.csv file.')
    ap.add_argument('-te', '--test_csv_path', help='Give the path to the test.csv file.')
    ap.add_argument('-nt', '--noise_type', help='type of label noise to be added: symmetry, flip or none')
    ap.add_argument('-nr', '--noise_rate', type=float, help='rate of label noise to be added, use float: between 0. and 1.')
    ap.add_argument('-ch', '--channels', default='RGB', help='Decide what channels you want to load: RGB or ALL. Default is RGB.')
    ap.add_argument('-lb', '--label_type', default='BigEarthNet-19', help='Decide what version of BigEarthNet you want to load: BigEarthNet-19 or original. Default is BigEarthNet-19.')
    ap.add_argument('-si', '--sigma', type=float, help='The value of the sigma for the gaussian kernel.')
    ap.add_argument('-sr', '--swap_rate', type=float, help='The percentage of the swap between the two models.')
    ap.add_argument('-lto', '--lambda_two', type=float, help='Lambda two for the L2.')
    ap.add_argument('-ltr', '--lambda_three', type=float, help='Lambda three for the L3.')
    args = vars(ap.parse_args())

    return args

# Path to save the model
PATH1 = './bigearth_dct_model1.pth'
PATH2 = './bigearth_dct_model2.pth'

# Save the model
def save_model(net1, net2):
    torch.save(net1.state_dict(), PATH)
    torch.save(net2.state_dict(), PATH)

def summarize_model(net1, net2):
    pass

def main(args):

    use_cuda = torch.cuda.is_available()
    print(f"use_cuda: {use_cuda}")

    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    if args['label_type'] == 'BigEarthNet-19':
        NUM_OF_CLASSES = 43
    elif args['label_type'] == 'original':
        NUM_OF_CLASSES = 19

    # Load the given dataset by user
    if args['lmdb_path']:  
        '''
        train_dataset = load_archive(args['dataset_path'] + '/train.tfrecord', NUM_OF_CLASSES, args['batch_size'], 1000)
        test_dataset = load_archive(args['dataset_path'] + '/test.tfrecord', NUM_OF_CLASSES, args['batch_size'], 1000)
        val_dataset = load_archive(args['dataset_path'] + '/val.tfrecord', NUM_OF_CLASSES, args['batch_size'], 1000)
        '''
        train_data_loader, val_data_loader = load_archive(args['lmdb_path'], args['train_csv_path'], args['val_csv_path'], args['test_csv_path'], args['batch_size'])
    else:
        raise ValueError('Argument Error: Give the path to the folder where the lmdb file is located.')
    
    # TODO: GET THE SHAPE FROM THE DATASET
    if args['channels'] == 'RGB':
        net1 = Net('SCNN_1', NUM_OF_CLASSES, args['batch_size'], args['epochs'], 3)
        net2 = Net('SCNN_2', NUM_OF_CLASSES, args['batch_size'], args['epochs'], 3)
    elif args['channels'] == 'ALL':
        net1 = Net('SCNN_1', NUM_OF_CLASSES, args['batch_size'], args['epochs'], 10)
        net2 = Net('SCNN_2', NUM_OF_CLASSES, args['batch_size'], args['epochs'], 10)
    else:
        raise ValueError('Argument Error: Legal arguments are RGB and ALL')
    
    if use_cuda:
        net1.cuda()
        net2.cuda()

    run_together(net1, net2, train_data_loader, val_data_loader, args['epochs'], args['batch_size'], args['sigma'], args['swap_rate'], args['lambda_two'], args['lambda_three'])
    
    summarize_model(net1, net2)

    # save_model(net1, net2)

if __name__ == '__main__':
    args = add_arguments()
    main(args)
