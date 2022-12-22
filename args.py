"""
Command-line argument parsing.
"""

import argparse
import time


# Used to parse boolean arguments
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
# args


train_actions = """directions,greeting,phoning,posing,purchases,sitting,sittingdown,takingphoto,waiting,walkingdog,walkingtogether"""

def argument_parser():
    """
    Get an argument parser for a training script.
    """
     
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name',  help='Name of the experiment', default="GraphHet", type=str)

    # Paths
    parser.add_argument('--data_dir',  help='A path pointing to the h3.6 data', default="dataset", type=str)
    parser.add_argument('--edges_dir',  help='A path pointing to the graph representing the h3.6 skeleton in .npy format', default="edges.npy", type=str) 
    parser.add_argument('--save_dir',  help='A path pointing to the folder you wish to save', default="logs", type=str)

    # Data loading
    parser.add_argument('--sample_joints',  help='Option to sample joints. Full samples all joints for the standard setup. Sub samples subgraphs from the original skeleton. Sample takes a subset of nodes and makes new connections', default="full", type=str)
    parser.add_argument('--min_joints',  help='Minimum amount of joints', default=5, type=int)
    parser.add_argument('--train_actions', help='actions for meta-training, defulat is all', default=train_actions, type=str)
    parser.add_argument('--max_joints',  help='Maximum amount of joints for sub', default=-1, type=int)

    # Training
    parser.add_argument('--learning_rate',  help='learning rate value', default=0.0001, type=float)
    parser.add_argument('--l2_rate',  help='l2 rate coefficient', default=0.0, type=float)
    parser.add_argument('--num_epochs',  help='number of epochs to train teh model', default=500, type=int)
    parser.add_argument('--steps',  help='number of steps in one epoch', default=50, type=int)
    

    # Model
    parser.add_argument('--dims_inf',       help='A list of model weights for inference network', default="[32,32,32]", type=str)
    parser.add_argument('--dims_pred',      help='A list of model weights for prediction network', default="[32,32,32]", type=str)

    parser.add_argument('--timehet_style',   help='select either the classical TimeHetNet (classic) or the graph one (graph) by default',   default="graph", type=str)
    parser.add_argument('--dropout',  help='Dropout rate', default=0.0, type=float)

    parser.add_argument('--mo_encoder',  help='Whether to use DSTC (True) or Conv Encoder (False)', default="True", type=boolean_string)

    return parser.parse_args()
