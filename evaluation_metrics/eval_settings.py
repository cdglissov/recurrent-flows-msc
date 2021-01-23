
import argparse
import torch
import sys
# Adding deepflows to system path
sys.path.insert(1, './deepflows/')

from evaluation_metrics.error_metrics import Evaluator

def main(settings):
    experiments = settings.experiment_names
    if settings.calc_eval:    
        for i in range(0,len(experiments)):
            model_name = settings.model_path[i]
            load_model = torch.load(settings.folder_path+experiments[i]+"/model_folder/"+model_name)
            args = load_model['args']
            
            if model_name == "svg.pt":
                from SVG.trainer import Solver
            elif model_name == "vrnn.pt":
                from VRNN.trainer import Solver
            elif model_name == "rfn.pt":
                from RFN.trainer import Solver
            elif model_name == "srnn.pt":
                from SRNN.trainer import Solver
            else:
                print("Unkown Model")
                
            solver = Solver(args)
            solver.build()
            solver.load(load_model)               
                			   			
            evaluator = Evaluator(solver, args, settings)
            evaluator.build()
            
            path_save_measures = settings.folder_path + experiments[i] + "/eval_folder"
            
            if not settings.test_temperature:
                evaluator.model.temperature = settings.temperatures[i]
                MSE_values, PSNR_values, SSIM_values, LPIPS_values = evaluator.get_eval_values()
                dict_values = {"SSIM_values": SSIM_values.cpu(),
                            "PSNR_values": PSNR_values.cpu(),
                            "MSE_values": MSE_values.cpu(),
                            "LPIPS_values": LPIPS_values.cpu(),
                            "temperature": settings.temperatures[i]
                            }
                
                torch.save(dict_values, path_save_measures + '/evaluations.pt')     
                print("SSIM:", SSIM_values.mean(0))
                print("PSNR:", PSNR_values.mean(0))
                print("MSE:", MSE_values.mean(0))
                print("LPIPS:", LPIPS_values.mean(0))
            else:
                for temperature in settings.temperatures:
                        evaluator.model.temperature = temperature
                        MSE_values, PSNR_values, SSIM_values, LPIPS_values = evaluator.get_eval_values()
                        dict_values = {"SSIM_values": SSIM_values.cpu(),
                                    "PSNR_values": PSNR_values.cpu(),
                                    "MSE_values": MSE_values.cpu(),
                                    "LPIPS_values": LPIPS_values.cpu(),
                                    "temperature": temperature
                                    }
                        torch.save(dict_values, path_save_measures + "/t" + str(temperature).replace('.','') + 'evaluations.pt')    

            
    # results always saved in the last eval_folder of experiment_names
    if not settings.test_temperature:
        evaluator.plot_eval_values(path = settings.folder_path,
                               label_names=settings.label_names, 
                               experiment_names=experiments)
    else:
        evaluator.test_temp_values(path = settings.folder_path,
                               label_names=settings.label_names, 
                               experiment_names=experiments)
 
    # FIX THIS FUNCTION
    #evaluator.loss_plots(path = settings.folder_path,
    #                     label_names=settings.label_names, 
    #                     experiment_names=experiments)
    
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
    
    # PATH SETTINGS:
    parser.add_argument("--folder_path", help="Path to folder that contains the experiments", 
                        default='./work1/s146996/', type=str)
    parser.add_argument("--experiment_names", nargs='+', help="Name of the experiments to eval",
                        default=["vrnn_test", "svg_test"], type=str)
    parser.add_argument("--model_path", nargs='+', help="Path to model.pt file", 
                        default=['vrnn.pt', 'svg.pt'], type=str)
    
    #CALCULATE VALUES SETTINGS:
    parser.add_argument("--num_samples_to_plot", help="This will create a plot of N sequences", 
                        default=5, type=int)
    parser.add_argument("--n_frames", help="Specify the sequence length of the test data", 
                        default=30, type=int)
    parser.add_argument("--start_predictions", help="Specify when model starts predicting", 
                        default=6, type=int)
    parser.add_argument('--temperatures', nargs='+', help="Specify temperature for the model", 
                        default=[0.6, 0.2], type=float)
    parser.add_argument("--resample", help="Loops over the test set more than once to get better measures. WARNING: can be slow", 
                        default=1, type=int)
    add_bool_arg(parser, "show_elbo_gap", default=False, 
                 help="Plots the elbo gap of the RFN model. WARNING: Only works for RFN")
    
    #TEST TEMPERATURE:
    add_bool_arg(parser, "test_temperature", default=True, 
                 help="Allows one to test temperature. If enabled different temperatures (from --temperatures) are tested for each specified model")
    
    #DEBUG SETTINGS:
    add_bool_arg(parser, "debug_mnist", default=True, 
                 help="Uses a small test set to speed up iterations for debugging. Only works for SM-MNIST")
    add_bool_arg(parser, "debug_plot", default=True, 
                 help="Plots num_samples_to_plot samples to make sure the loader and eval works")
    
    #EVAL VALUES PLOTTER SETTINGS:
    add_bool_arg(parser, "calc_eval", default=True, 
                 help="Set to false if we do not want to calculate eval values")
    parser.add_argument("--n_conditions", help="Number of conditions used for plotting eval_values", 
                        default=1, type=int) #TODO: Ask tobs about this, isn't this equal to start_predictions?
    parser.add_argument("--label_names", nargs='+', help="Name of the labels for the eval plots",
                        default=["VRNN", "SVG", " SRNN"], type=str)
    
    #LOSS VALUES PLOTTER SETTINGS:
    #TODO: fix this function
    
    args = parser.parse_args()
    
    main(args)