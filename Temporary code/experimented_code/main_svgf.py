import argparse
from SVG_Flow.trainer import Solver
from torch import load as tloader

def main(args):
    
    if args.load_model:
        load_model = tloader(args.path + 'model_folder/rfn.pt')
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
                        default=60, type=int)
    parser.add_argument("--n_frames", help="Specify number of frames", 
                        default=10, type=int)
    parser.add_argument("--n_past", help="Specify number of conditioned frames", 
                        default=6, type=int)
    parser.add_argument("--n_future", help="Specify number of predicted future frames", 
                        default=4, type=int)
    parser.add_argument("--choose_data", help="Specify dataset", 
                        choices=['mnist', 'bair'], default='mnist', type=str)
    parser.add_argument("--image_size", help="Specify the image size of mnist", 
                        default=64, type=int)
    parser.add_argument("--digit_size", help="Specify the size of mnist digit", 
                        default=32, type=int)
    parser.add_argument("--step_length", help="Specify the step size of mnist digit", 
                        default=4, type=int)
    parser.add_argument("--num_digits", help="Specify the number of mnist digits", 
                        default=2, type=int)
    parser.add_argument("--num_workers", help="Specify the number of workers in dataloaders", 
                        default=2, type=int)
    
    
    # Trainer
    parser.add_argument("--patience_es", help="Specify patience for early stopping", 
                        default=500, type=int)
    parser.add_argument("--patience_lr", help="Specify patience for lr_scheduler", 
                        default=500, type=int)
    parser.add_argument("--factor_lr", help="Specify lr_scheduler factor (0..1)", 
                        default=0.5, type=restricted_float)
    parser.add_argument("--min_lr", help="Specify minimum lr for scheduler", 
                        default=0.00001, type=float)
    parser.add_argument("--n_bits", help="Specify number of bits", 
                        default=6, type=int)
    parser.add_argument("--n_epochs", help="Specify number of epochs", 
                        default=100, type=int)
    add_bool_arg(parser, "verbose", default=False, help="Specify verbose mode (boolean)")
    parser.add_argument("--path", help="Specify path to experiment", 
                        default='/content_svg/', type=str)
    parser.add_argument("--preprocess_range", help="Specify the range of the data for preprocessing", 
                        choices=['0.5','1.0'], default='0.5', type=str)
    parser.add_argument("--preprocess_scale", help="Specify the scale for preprocessing", 
                        default=255, type=int)
    parser.add_argument("--beta_max", help="Specify the maximum value of beta", 
                        default=1, type=float)
    parser.add_argument("--beta_min", help="Specify the minimum value of beta", 
                        default=0.000001, type=float)
    parser.add_argument("--beta_steps", help="Specify the annealing steps", 
                        default=5000, type=int)
    parser.add_argument("--n_predictions", help="Specify number of predictions", 
                        default=6, type=int)
    add_bool_arg(parser, "multigpu", default=False, help="Specify if we want to use multi GPUs")
    add_bool_arg(parser, "load_model", default=False, 
                 help="Specify if we want to load a pre-existing model (boolean)")
    add_bool_arg(parser, "variable_beta", default=True, 
                 help="Specify if we want a linear decreasing beta value (boolean)")
    parser.add_argument("--optimizer", help="Specify the type of optimizer", 
                        choices=['rmsprop','adam'], default='adam', type=str)
    
    
    parser.add_argument('--x_dim', nargs='+', help="Specify data dimensions (b,c,h,w)", 
                        default=[60, 1, 64, 64], type=int)
    parser.add_argument('--condition_dim', nargs='+', help="Specify condition dimensions (b,c,h,w)", 
                        default=[60, 1, 64, 64], type=int)
    parser.add_argument("--h_dim", help="Specify hidden state (h) channels", 
                        default=256, type=int)
    parser.add_argument("--z_dim", help="Specify latent (z) channels", 
                        default=20, type=int)
    parser.add_argument("--L", help="Specify flow depth", 
                        default=6, type=int)
    parser.add_argument("--K", help="Specify flow recursion", 
                        default=16, type=int)
    parser.add_argument("--c_features", help="Specify channels of extracted features", 
                        default=256, type=int)
    parser.add_argument("--temperature", help="Specify temperature", 
                        default=0.8, type=restricted_float)
    parser.add_argument("--posterior_rnn_layers", help="Specify layers of posterier (variational encoder)", 
                        default=1, type=int)
    parser.add_argument("--predictor_rnn_layers", help="Specify layers of predictor", 
                        default=2, type=int)
    parser.add_argument("--prior_rnn_layers", help="Specify layers of variational prior", 
                        default=1, type=int)
    parser.add_argument("--betas_low", help="Specify running average of gradient", 
                        default=0.9, type=restricted_float)
    parser.add_argument("--learning_rate", help="Specify learning_rate", 
                        default=0.0001, type=float)
    parser.add_argument("--free_bits", help="Specify free bit", 
                        default=0.0, type=restricted_float)
    parser.add_argument('--act_upscaler', help="Specify activation for SVG upscaler", 
                        default='tanh', choices=["tanh", "sigmoid", "none"], type=str)
    #Glow
    add_bool_arg(parser, "learn_prior", default=True, help="Specify if we want a learned prior (boolean)")
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
                        default='realnvp', choices=["glow", "realnvp", "softclamp"], type=str)
    
    args = parser.parse_args()
    
    
    main(args)
