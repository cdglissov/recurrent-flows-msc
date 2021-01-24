# TODO: check metric STD's
# TODO: Ask tobs about if n_conditions isn't equal to start_predictions? (it is)
# TODO: Proper normalized loss in plot_elbo_gap
# TODO: Make FVD wrapper in error_metrics.py https://github.com/edouardelasalles/srvp/tree/master/metrics
# TODO: Get parameters of RFN model (Tomorrow)
# TODO: get best samples and PSNR values (laver jeg i dag) best_predictions, worst_predictions ud fra SSIM, find idividuelle sequences
# TODO: Avg. bits per dim over frames (reconstructPlus) (Tobias)
# TODO: Interpolation between two images (Tobias)
# TODO: Kvantiser hvor god modellen er til at reconstruct (Tobias) + amortization gap

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
                MSE_values, PSNR_values, SSIM_values, LPIPS_values, BPD, DKL, RECON = evaluator.get_eval_values(model_name)
                dict_values = {"SSIM_values": SSIM_values.cpu(),
                            "PSNR_values": PSNR_values.cpu(),
                            "MSE_values": MSE_values.cpu(),
                            "LPIPS_values": LPIPS_values.cpu(),
                            "temperature": settings.temperatures[i],
                            "BPD": BPD.cpu(),
                            "DKL": DKL.cpu(),
                            "RECON": RECON.cpu(),
                            }
                
                torch.save(dict_values, path_save_measures + '/evaluations.pt')     
                
                with open(path_save_measures+'/eval_avg_losses.txt', 'w') as f:
                    print("SSIM:", SSIM_values.mean(0), file=f)
                    print("PSNR:", PSNR_values.mean(0), file=f)
                    print("MSE:", MSE_values.mean(0), file=f)
                    print("LPIPS:", LPIPS_values.mean(0), file=f)
                    print("BPD:", BPD.mean(), file=f)
                    print("DKL:", DKL.mean(),file=f)
                    print("RECON:", RECON.mean(),file=f)

            else:
                for temperature in settings.temperatures:
                        evaluator.model.temperature = temperature
                        MSE_values, PSNR_values, SSIM_values, LPIPS_values = evaluator.get_eval_values(model_name)
                        dict_values = {"SSIM_values": SSIM_values.cpu(),
                                    "PSNR_values": PSNR_values.cpu(),
                                    "MSE_values": MSE_values.cpu(),
                                    "LPIPS_values": LPIPS_values.cpu(),
                                    "temperature": temperature,
                                    "BPD": BPD.cpu(),
                                    "DKL": DKL.cpu(),
                                    "RECON": RECON.cpu(),
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
                        default=["rfn_test", "svg_test"], type=str)
    parser.add_argument("--model_path", nargs='+', help="Name of model.pt file", 
                        default=['rfn.pt', "svg.pt"], type=str)
    
    #CALCULATE VALUES SETTINGS:
    parser.add_argument("--num_samples_to_plot", help="This will create a plot of N sequences", 
                        default=5, type=int)
    parser.add_argument("--n_frames", help="Specify the sequence length of the test data", 
                        default=30, type=int)
    parser.add_argument("--start_predictions", help="Specify when model starts predicting", 
                        default=6, type=int)
    parser.add_argument('--temperatures', nargs='+', help="Specify temperature for the model", 
                        default=[0.6], type=float)
    parser.add_argument("--resample", help="Loops over the test set more than once to get better measures. WARNING: can be slow", 
                        default=1, type=int)
    add_bool_arg(parser, "show_elbo_gap", default=False, 
                 help="Plots the elbo gap of the RFN model. WARNING: Only works for RFN")
    
    #TEST TEMPERATURE:
    add_bool_arg(parser, "test_temperature", default=False, 
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
                        default=6, type=int)
    parser.add_argument("--label_names", nargs='+', help="Name of the labels for the eval plots",
                        default=["RFN"], type=str)
    
    
    args = parser.parse_args()
    
    main(args)