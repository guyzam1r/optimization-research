import numpy as np
import pandas as pd
import torch
from torch import nn
from matplotlib import pyplot as plt
import time

from optim import *
from params import *

"""
Run expirements using the given optimizers and parameters. Returns a Pandas DataFrame with the labels, losses, times, and epochs.

Args
    obj: The objective function (e.g. rosenbrock).
    opts: The optimizers, passed in as a list of strings (e.g. ["gd", "gd", "adam", "adam"]).
    epochs: The number of epochs to run for.
    x_init: A one-dimensional pytorch tensor representing the initial weights that we will train.
    labels: A list of strings, where each string is the label for the corresponding optimizer.
    param_changes:
    tel_scheds: A list of lists, where each list contains the epochs in which we teleport for the corresponding optimizer.
    figsz: The figure size. The default is (12,8).
    ylim: The upper limit of the y-axis for the plot. The default is -1, which sets the limit automatically.
    multiruns: The defult is [], which runs everything once. Can only use when use_time is False.
    verbose: The default is 1. If set to 0, prints nothing. If set to 2, the weights will be printed following each epoch.
"""
def run(obj, opts, duration, x_init, labels, param_changes, tel_scheds, use_time=False, multiruns=[], verbose=1):
    #Set default value of multiruns
    if multiruns == []:
        multiruns = [1] * len(opts)
    elif use_time:
        print("Cannot use multiruns with time duration")
        return
        
    dfs = []
    
    for i in range(len(opts)):
        opt = opts[i]
        tel_sched = tel_scheds[i]
        cur_changes = param_changes[i]
        runs = multiruns[i]
        
        if not use_time:
            total_loss_vals = np.zeros(duration+1)
            
        for run in range(runs):
            x_init_clone = x_init.clone().detach().requires_grad_(True)
            parameters = set_default_params(x_init_clone.size(0))
            #set_torch_optimizers(x_init_clone)

            #modify parameters for current round of training
            for change in cur_changes:
                parameters = set_param(parameters, change[0], change[1], change[2])

            #train, then store losses
            x, loss_vals, times = train(obj, opt, duration, x_init_clone, parameters, tel_sched, use_time, verbose)
            
            if (not use_time) and (runs > 1):
                total_loss_vals += np.array(loss_vals)
        
        if (not use_time) and (runs > 1):
            #take average of the losses over all runs
            loss_vals = total_loss_vals / runs

        #Save data
        cur_df = pd.DataFrame({
            "label":[labels[i]] * len(loss_vals),
            "epoch":np.arange(len(loss_vals)),
            "loss":loss_vals,
            "time":times
        })
        dfs.append(cur_df)
            
        if verbose >= 1:        
            print("Total time elapsed using", labels[i], "-", times[-1])
            print("Final x -", x)
            print("Final loss -", obj(x).item())
            #if obj == multi_layer:
            #    print("Final score:", test_score(x))
            print("------------------------------")

    return pd.concat(dfs)

"""
Optimize an objective for a certain number of epochs or seconds.

Args
    obj: The objective function (e.g. rosenbrock).
    opt: The optimizer, passed in as a string (e.g. "adam").
    duration: If use_time is set to False (default), this is the number of epochs to run for. 
              If use_time is set to True, this is the number of seconds to run for.
    x: The initial weights that we will train.
    teleport_sched: A list containing the epochs in which we teleport.
    use_time: Set True for duration to be in seconds, False (default) for duration to be in epochs.
    verbose: The default is 1. If set to 0, prints nothing. If set to 2, the weights will be printed following each epoch.

Returns
    x: The final weights.
    loss_vals: A list containing the loss after each epoch.
    times: A list containing the total time expired (since the start of training) after each epoch.
"""
def train(obj, opt, duration, x, params, teleport_sched, use_time=False, verbose=1):
    loss_vals = [obj(x).item()]
    times = [0]
    
    start = time.time()

    #Duration in seconds
    if use_time:
        epoch = 0
        while time.time()-start < duration:
            if epoch in teleport_sched:
                x, params = step(opt, params, x, obj, True)
                if verbose == 2:
                    print("Teleporting now!")
            else:
                x, params = step(opt, params, x, obj, False)
                
            if verbose == 2:
                print(x)

            loss_vals.append(obj(x).item())
            times.append(time.time() - start)
            epoch += 1
    
    #Duration in epochs
    else:
        for epoch in range(duration):
            if epoch in teleport_sched:
                x, params = step(opt, params, x, obj, True)
                if verbose == 2:
                    print("Teleporting now!")
            else:
                x, params = step(opt, params, x, obj, False)
                
            if verbose == 2:
                print(x)

            loss_vals.append(obj(x).item())
            times.append(time.time() - start)
        
    return x, loss_vals, times