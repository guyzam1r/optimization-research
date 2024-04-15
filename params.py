import numpy as np
import torch
from torch import nn

def set_param(params, opt, param, new_value):
    params[opt][param] = new_value
    return params

def set_default_params(n): 
    gd_params = {
        "lr": 0.002
    }

    momentum_params = {
        "lr": 0.002,
        "v": 0,
        "decay": 0.5,
        "nesterov": False,
        "scaling": 0,
        "scale_type": 0
    }

    adagrad_params = {
        "lr": 1,
        "s": 0
    }

    adam_params = {
        "lr": 0.08,
        "v": 0,
        "s": 0,
        "v_decay": 0.9,
        "s_decay": 0.999,
        "k": 0,
        "tel_reset": False
    }
    
    #OLD/NEW CMAES PARAMETERS
    l = int(4 + np.floor(3*np.log(n)))
    mu = int(np.floor(l/4))
    #l = 200
    #mu = 10
    chi = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
    acov = 1/mu
    
    oldcmaes_params = {
        "l": l,
        "mu": mu,
        "C": torch.eye(n),
        "pc": torch.zeros(n),
        "cc": 4 / (n+4),
        "ccov": 2 / (n + np.sqrt(2))**2,
        "s": 0.1,
        "ps": torch.zeros(n),
        "cs": 4/(n+4),
        "ds": 1 + (n+4)/4,
        "chi": chi
    }
    
    newcmaes_params = {
        "l": l,
        "mu": mu,
        "C": torch.eye(n),
        "pc": torch.zeros(n),
        "cc": 4 / (n+4),
        "ccov": acov*2/(n+np.sqrt(2))**2 + (1-acov)*min(1, (2*mu-1)/((n+2)**2 + mu)),
        "acov": acov,
        "s": 0.01,
        "ps": torch.zeros(n),
        "cs": 4/(n+4),
        "ds": 1 + (n+4)/4,
        "chi": chi
    }
    
    #CMAES PARAMETERS
    l = int(4 + np.floor(3 * np.log(n)))
    mu = int(np.floor(l/2))
    
    w_prime = np.log((l+1)/2) - torch.log(torch.arange(1,l+1))
        
    mu_eff = (torch.sum(w_prime[:mu])**2) / torch.sum(w_prime[:mu]**2)
    mu_eff_neg = (torch.sum(w_prime[mu:])**2) / torch.sum(w_prime[mu:]**2)
    
    cm = 1
    
    cs = (mu_eff + 2) / (n + mu_eff + 5)
    ds = 1 + 2 * max(0, np.sqrt((mu_eff-1)/(n+1))-1) + cs
    
    cc = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
    c1 = 2 / ((n+1.3)**2 + mu_eff)
    cmu = min(1-c1, 2*(1/4 + mu_eff + 1/mu_eff - 2)/((n+2)**2 + mu_eff))
    
    amu = 1 + c1/cmu
    amu_eff = 1 + (2*mu_eff_neg) / (mu_eff + 2)
    a_pd = (1 - c1 - cmu) / (n*cmu)
    
    sum_w_prime_pos = torch.sum(w_prime[:mu])
    sum_w_prime_neg = -torch.sum(w_prime[mu:])
    
    w = torch.zeros(l)
    w[:mu] = w_prime[:mu] / sum_w_prime_pos
    w[mu:] = min(amu, amu_eff, a_pd) * w_prime[mu:] / sum_w_prime_neg
    sumw = torch.sum(w)
    
    chi = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
    
    cmaes_params = {
        "l": l,
        "mu": mu,
        "mu_eff": mu_eff,
        "C": torch.eye(n),
        "pc": torch.zeros(n),
        "cc": cc,
        "cm": cm,
        "c1": c1,
        "cmu": cmu,
        "s": 0.1,
        "ps": torch.zeros(n),
        "cs": cs,
        "ds": ds,
        "chi": chi,
        "w": w,
        "sumw": sumw,
        "epoch": 0
    }

    parameters = {
        "gd": gd_params,
        "momentum": momentum_params,
        "adagrad": adagrad_params,
        "adam": adam_params,
        "oldcmaes": oldcmaes_params,
        "newcmaes": newcmaes_params,
        "cmaes": cmaes_params
    }
    
    return parameters


def set_torch_optimizers(x):
    global adamopt
    adamopt = torch.optim.Adam([x], lr=0.08)