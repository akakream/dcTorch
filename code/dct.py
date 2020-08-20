import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    ap.add_argument('-d', '--dataset_path', help='Give the path to the folder where tf.record files are located.')
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

    if args['framework'] not in ('co', 'single'): 
        raise ValueError('Enter "single" for a single model, enter "co" for collaborative model in the framework argument.')
    
    if args['label_type'] == 'BigEarthNet-19':
        NUM_OF_CLASSES = 43
    elif args['label_type'] == 'original':
        NUM_OF_CLASSES = 19

    # Load the given dataset by user
    if args['dataset_path']:  
        '''
        train_dataset = load_archive(args['dataset_path'] + '/train.tfrecord', NUM_OF_CLASSES, args['batch_size'], 1000)
        test_dataset = load_archive(args['dataset_path'] + '/test.tfrecord', NUM_OF_CLASSES, args['batch_size'], 1000)
        val_dataset = load_archive(args['dataset_path'] + '/val.tfrecord', NUM_OF_CLASSES, args['batch_size'], 1000)
        '''
        train_dataset = 0
        test_dataset = 0
        val_dataset = 0
    else:
        raise ValueError('Argument Error: Give the path to the folder where tf.record files are located.')
    
    # TODO: GET THE SHAPE FROM THE DATASET
    if args['channels'] == 'RGB':
        net1 = Net('SCNN_1', NUM_OF_CLASSES, args['batch_size'], args['epochs'], 3)
        net2 = Net('SCNN_2', NUM_OF_CLASSES, args['batch_size'], args['epochs'], 3)
    elif args['channels'] == 'ALL':
        net1 = Net('SCNN_1', NUM_OF_CLASSES, args['batch_size'], args['epochs'], 12)
        net2 = Net('SCNN_2', NUM_OF_CLASSES, args['batch_size'], args['epochs'], 12)
    else:
        raise ValueError('Argument Error: Legal arguments are RGB and ALL')

    run_together(net1, net2, train_dataset, test_dataset, val_dataset, args['epochs'], args['batch_size'], args['sigma'], args['swap_rate'], args['lambda_two'], args['lambda_three'])
    
    summarize_model(net1, net2)

    # save_model(net1, net2)

    '''
    # INPUT                                                                         
    inp = torch.randn(1,3,120,120)                                                  
    output = net(inp)                                                               
    print(output) 
    '''


if __name__ == '__main__':
    args = add_arguments()
    main(args)

