#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from cFlow.trainer import Solver
from torch import load as tloader

def main(args):
    
    if args.load_model:
        load_model = tloader('.'+args.path + 'model_folder/rfn.pt')
        args = load_model['args']
        solver = Solver(args)
        solver.build()
        solver.load(load_model)
    else:
        solver = Solver(args)
        solver.build()
        
    solver.train()   

    
def add_bool_arg(parser, name, help, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=name, action='store_false', help=help)
    parser.set_defaults(**{name:default})


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


def convert_mixed_list(x):
    if x.isdigit():
        return int(x)
    else:
        return x


def convert_to_upscaler(x):
    block = [convert_mixed_list(i) for i in x.split("-")]
    return block

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    #DATA
    parser.add_argument("--batch_size", help="Specify batch size", 
                        default=40, type=int)
    parser.add_argument("--n_frames", help="Specify number of frames", 
                        default=6, type=int)
    parser.add_argument("--choose_data", help="Specify dataset", 
                        choices=['celeba32'], default='celeba32', type=str)
    parser.add_argument("--image_size", help="Specify the image size of mnist", 
                        default=64, type=int)
    parser.add_argument("--num_workers", help="Specify the number of workers in dataloaders", 
                        default=2, type=int)
    
    
    # Trainer
    parser.add_argument("--patience_es", help="Specify patience for early stopping", 
                        default=5000, type=int)
    parser.add_argument("--patience_lr", help="Specify patience for lr_scheduler", 
                        default=3, type=int)
    parser.add_argument("--factor_lr", help="Specify lr_scheduler factor (0..1)", 
                        default=0.3, type=restricted_float)
    parser.add_argument("--min_lr", help="Specify minimum lr for scheduler", 
                        default=0.00001, type=float)
    parser.add_argument("--n_bits", help="Specify number of bits", 
                        default=5, type=int)
    parser.add_argument("--n_epochs", help="Specify number of epochs", 
                        default=100000, type=int)
    add_bool_arg(parser, "verbose", default=False, help="Specify verbose mode (boolean)")
    parser.add_argument("--path", help="Specify path to experiment", 
                        default='/content/', type=str)
    parser.add_argument("--learning_rate", help="Specify learning_rate", 
                        default=0.0001, type=float)
    parser.add_argument("--preprocess_range", help="Specify the range of the data for preprocessing", 
                        choices=['0.5','1.0'], default='0.5', type=str)
    parser.add_argument("--preprocess_scale", help="Specify the scale for preprocessing", 
                        default=255, type=int)

    parser.add_argument("--n_predictions", help="Specify number of predictions", 
                        default=6, type=int)
    add_bool_arg(parser, "multigpu", default=False, help="Specify if we want to use multi GPUs")
    add_bool_arg(parser, "load_model", default=False, 
                 help="Specify if we want to load a pre-existing model (boolean)")
    parser.add_argument('--norm_type_features', help="Specify normalization type of layers upscaler/downscaler", 
                    default='batchnorm', choices=["instancenorm", "batchnorm", "none"], type=str)
    
    # RFN
    parser.add_argument('--x_dim', nargs='+', help="Specify data dimensions (b,c,h,w)", 
                        default=[40, 1, 64, 64], type=int)
    parser.add_argument('--condition_dim', nargs='+', help="Specify condition dimensions (b,c,h,w)", 
                        default=[40, 1, 64, 64], type=int)
    parser.add_argument("--L", help="Specify flow depth", 
                        default=3, type=int)
    parser.add_argument("--K", help="Specify flow recursion", 
                        default=10, type=int)
    # Downscaler architechture. Consisting of L blocks, the end of each block
    # will be what is skip connected. It is possible to chose between 
    # pool,conv,squeeze, as downsampling methods. Does not need to end with integer
    parser.add_argument('--extractor_structure', nargs='+', help="Specify structure of extractor example writing, 32-32-conv 32-32-pool, creates 2 blocks", 
                        default= [[16, 16, 'squeeze'],[32, 32, 'squeeze'], [64, 64, 'squeeze']], type=convert_to_upscaler)

    add_bool_arg(parser, "downscaler_tanh", default=False, help="Specify if skip connection from downscaler is tanh'ed (boolean)")
    parser.add_argument("--structure_scaler", help="Specify down/up-sampling channel factor", 
                        default=2, type=int)
    parser.add_argument("--temperature", help="Specify temperature", 
                        default=0.8, type=restricted_float)
    add_bool_arg(parser, "skip_connection_features", default=False, help="Specify if skip connection between up and downscaler (boolean)")
    
    add_bool_arg(parser, "learn_prior", default=True, help="Specify if conditional prior or not")
    

    add_bool_arg(parser, "one_condition", default=False, help="If true, condition will have the same spatial size for every layer of the multiscale struckture. If false the codntion will be scaled to match the flow, in conditioner net.")
    
    add_bool_arg(parser, "LU_decomposed", default=True, help="Specify if we want to use LU factorization (boolean)")
    parser.add_argument("--n_units_affine", help="Specify hidden units in affine coupling", 
                        default=256, type=int)
    parser.add_argument("--non_lin_glow", help="Specify activation in glow", 
                        default="relu", choices=["relu", "leakyrelu"], type=str)
    parser.add_argument("--n_units_prior", help="Specify hidden units in prior", 
                        default=256, type=int)
    add_bool_arg(parser, "make_conditional", default=True, 
                 help="Specify if split should be conditional or not (boolean)")
    parser.add_argument('--flow_norm', help="Specify normalization type of glow-step", 
                        default='actnorm', choices=["batchnorm", "actnorm"], type=str)
    parser.add_argument('--base_norm', help="Specify normalization type of base distribution", 
                        default='actnorm', choices=["batchnorm", "actnorm"], type=str)
    parser.add_argument("--flow_batchnorm_momentum", help="Running average batchnorm momentum for flow-step", 
                        default=0.0, type=float)
    parser.add_argument('--clamp_type', help="Specify clamp type of affine coupling", 
                        default='realnvp', choices=["glow", "realnvp", "softclamp", "none"], type=str)
    parser.add_argument('--split2d_act', help="Specify clamp type of split2d", 
                        default='softplus', choices=["softplus", "exp"], type=str)
    args = parser.parse_args()
    
    main(args)

