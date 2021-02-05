import argparse
from SVG.trainer import Solver
from torch import load as tloader

def main(args):
    if args.load_model:
        load_model = tloader('.'+args.path + 'model_folder/svg.pt')
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
                        default=100, type=int)
    parser.add_argument("--n_frames", help="Specify number of frames", 
                        default=10, type=int)
    parser.add_argument("--choose_data", help="Specify dataset", 
                        choices=['mnist', 'bair', 'kth'], default='mnist', type=str)
    parser.add_argument("--image_size", help="Specify the image size of mnist", 
                        default=64, type=int)
    parser.add_argument("--digit_size", help="Specify the size of mnist digit", 
                        default=32, type=int)
    parser.add_argument("--step_length", help="Specify the step size of mnist digit", 
                        default=4, type=int)
    parser.add_argument("--num_digits", help="Specify the number of mnist digits", 
                        default=2, type=int)
    parser.add_argument("--num_workers", help="Specify the number of workers in dataloaders", 
                        default=4, type=int)
    add_bool_arg(parser, "use_validation_set", default=False, help="Specify if we want to use a validation set")
    
    # Trainer
    parser.add_argument("--patience_es", help="Specify patience for early stopping", 
                        default=50, type=int)
    parser.add_argument("--patience_lr", help="Specify patience for lr_scheduler", 
                        default=20, type=int)
    parser.add_argument("--factor_lr", help="Specify lr_scheduler factor (0..1)", 
                        default=0.5, type=restricted_float)
    parser.add_argument("--min_lr", help="Specify minimum lr for scheduler", 
                        default=0.0001, type=float)
    parser.add_argument("--n_bits", help="Specify number of bits", 
                        default=8, type=int)
    parser.add_argument("--n_epochs", help="Specify number of epochs", 
                        default=200, type=int)
    add_bool_arg(parser, "verbose", default=False, help="Specify verbose mode (boolean)")
    parser.add_argument("--path", help="Specify path to experiment", 
                        default='/content/', type=str)
    parser.add_argument("--learning_rate", help="Specify learning_rate", 
                        default=0.001, type=float)
    parser.add_argument("--preprocess_range", help="Specify the range of the data for preprocessing", 
                        choices=['0.5','1.0', 'minmax', 'none'], default='none', type=str)
    parser.add_argument("--preprocess_scale", help="Specify the scale for preprocessing", 
                        default=255, type=int)
    parser.add_argument("--beta_max", help="Specify the maximum value of beta", 
                        default=0.0001, type=float)
    parser.add_argument("--beta_min", help="Specify the minimum value of beta", 
                        default=0.0001, type=float)
    parser.add_argument("--beta_steps", help="Specify the annealing steps", 
                        default=1, type=int)
    parser.add_argument("--n_predictions", help="Specify number of predictions", 
                        default=5, type=int)
    parser.add_argument("--n_conditions", help="Specify number of conditions", 
                        default=5, type=int)
    add_bool_arg(parser, "multigpu", default=False, help="Specify if we want to use multi GPUs")
    add_bool_arg(parser, "load_model", default=False, 
                 help="Specify if we want to load a pre-existing model (boolean)")

    
    parser.add_argument("--posterior_rnn_layers", help="Specify layers of posterier (variational encoder)", 
                        default=1, type=int)
    parser.add_argument("--predictor_rnn_layers", help="Specify layers of predictor", 
                        default=2, type=int)
    parser.add_argument("--prior_rnn_layers", help="Specify layers of variational prior", 
                        default=1, type=int)
    parser.add_argument("--c_features", help="Specify channels of extracted features", 
                        default=256, type=int)
    parser.add_argument('--x_dim', nargs='+', help="Specify data dimensions (b,c,h,w)", 
                        default=[100, 1, 64, 64], type=int)
    parser.add_argument('--condition_dim', nargs='+', help="Specify condition dimensions (b,c,h,w)", 
                        default=[100, 1, 64, 64], type=int)
    parser.add_argument("--h_dim", help="Specify hidden state (h) channels", 
                        default=256, type=int)
    parser.add_argument("--z_dim", help="Specify latent (z) channels", 
                        default=12, type=int) # Very bad if too high, very bad if too low
    parser.add_argument("--loss_type", help="Specify the type of loss used", 
                        default="mse", choices = ["bernoulli", "mse", "gaussian"], type=str)
    parser.add_argument("--norm_type", help="Specify the type of loss used", 
                        default="batchnorm", choices = ["batchnorm", "instancenorm", "none"], type=str)
    parser.add_argument("--variance", help="Specify the variance of out put probability ", 
                        default=1, type=float)
    
    args = parser.parse_args()
    
    main(args)