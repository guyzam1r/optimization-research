import numpy as np
import torch
from torch import nn

from objectives import *
from teleport import *
from optim import *

def step(optimizer, params, x, obj, teleport):
    tel_dist = 0
    if teleport:
        if obj == booth:
            x, tel_dist = teleport_SO2(x, booth_x_to_v, booth_v_to_x, obj, 0.001)
        elif obj == rosenbrock:
            x, tel_dist = teleport_SO2(x, ros_x_to_v, ros_v_to_x, obj, 0.001)
        elif obj == sphere:
            x, tel_dist = teleport_SO2(x, sphere_x_to_v, sphere_v_to_x, obj, 0.01)
        elif obj == ellipsoid:
            x, tel_dist = teleport_SO2(x, elli_x_to_v, elli_v_to_x, obj, 0.01)
       #elif obj == multi_layer:
       #     x, tel_dist = teleport_MLP(x, X, Y, 1e-4, w_dims, obj)
       #     print(tel_dist)
        else:
            print("WARNING: CANNOT TELEPORT ON", obj)
    
    if optimizer == "gd":
        x_new = gd_step(x, obj, params["gd"])
        
    elif optimizer == "momentum":
        x_new, v_new = momentum_step(x, obj, tel_dist, params["momentum"])
        params["momentum"]["v"] = v_new
        
    elif optimizer == "adagrad":
        x_new, s_new = adagrad_step(x, obj, params["adagrad"])
        params["adagrad"]["s"] = s_new
        
    elif optimizer == "rmsprop":
        pass
    
    elif optimizer == "adam":
        x_new, v_new, s_new, k = adam_step(x, obj, tel_dist, params["adam"])
        params["adam"]["v"] = v_new
        params["adam"]["s"] = s_new
        params["adam"]["k"] = k
        
    #elif optimizer == "torchadam":
    #    torchadam_step(x, obj)
    #    x_new = x
    
    elif optimizer == "oldcmaes":
        x_new, C_new, pc_new, s_new, ps_new = oldcmaes_step(x, obj, tel_dist, params["oldcmaes"])
        params["oldcmaes"]["C"] = C_new
        params["oldcmaes"]["pc"] = pc_new
        params["oldcmaes"]["s"] = s_new
        params["oldcmaes"]["ps"] = ps_new
        
    elif optimizer == "newcmaes":
        x_new, C_new, pc_new, s_new, ps_new = newcmaes_step(x, obj, tel_dist, params["newcmaes"])
        params["newcmaes"]["C"] = C_new
        params["newcmaes"]["pc"] = pc_new
        params["newcmaes"]["s"] = s_new
        params["newcmaes"]["ps"] = ps_new
        
    elif optimizer == "cmaes":
        #x_new, C_new, pc_new, s_new, ps_new = cmaes_step(x, obj, tel_dist, params["cmaes"])
        x_new, C_new, pc_new, s_new, ps_new = cmaes_step(x, obj, params["cmaes"])
        params["cmaes"]["C"] = C_new
        params["cmaes"]["pc"] = pc_new
        params["cmaes"]["s"] = s_new
        params["cmaes"]["ps"] = ps_new
        params["cmaes"]["epoch"] = params["cmaes"]["epoch"] + 1
    
    else:
        print("Optimizer", optimizer, "not found.")
        
    return x_new, params




#Source: https://algorithmsbook.com/optimization/files/optimization.pdf

#Gradient Descent
def gd_step(x, obj, params):
    lr = params["lr"]
    
    loss = obj(x)
    grad, = torch.autograd.grad(loss, inputs=x)
    x_new = x - lr*grad
    return x_new

#Momentum
def momentum_step(x, obj, tel_dist, params):
    lr = params["lr"]
    v = params["v"]
    decay = params["decay"]
    nesterov = params["nesterov"]
    scaling = params["scaling"]
    scale_type = params["scale_type"]
    
    if scale_type > 0 and tel_dist > 0:
        if scale_type == 1:
            decay = decay * scaling
        if scale_type == 2:
            print("TEL DIST:", tel_dist)
            print("SCALE FACT:", torch.exp(-tel_dist))
            decay = decay * scaling * torch.exp(-tel_dist)
    
    if nesterov:
        proj_x = x + decay*v
        loss = obj(proj_x)
        grad, = torch.autograd.grad(loss, inputs=proj_x)
    else:
        loss = obj(x)
        grad, = torch.autograd.grad(loss, inputs=x)
    v_new = decay*v - lr*grad
    x_new = x + v_new
    return x_new, v_new

#AdaGrad
def adagrad_step(x, obj, params):
    lr = params["lr"]
    s = params["s"]
    
    loss = obj(x)
    grad, = torch.autograd.grad(loss, inputs=x)
    s_new = s + grad**2
    x_new = x - (lr / (10e-8 + torch.sqrt(s_new)))*grad
    return x_new, s_new

#Adam
def adam_step(x, obj, tel_dist, params):
    v = params["v"]
    v_decay = params["v_decay"]
    s = params["s"]
    s_decay = params["s_decay"]
    lr = params["lr"]
    k = params["k"] #stores number of iterations
    tel_reset = params["tel_reset"]
    
    if tel_dist > 0 and tel_reset:
        k = 0
        v = 0
        s = 0
    
    k += 1
    
    loss = obj(x)
    grad, = torch.autograd.grad(loss, inputs=x)
    
    v_new = v_decay*v + (1-v_decay)*grad
    s_new = s_decay*s + (1-s_decay)*(grad**2)
    
    v_hat = v_new / (1 - v_decay**k)
    s_hat = s_new / (1 - s_decay**k)
    
    x_new = x - lr * v_hat / (1e-8 + torch.sqrt(s_hat))
    return x_new, v_new, s_new, k

"""
def torchadam_step(x, obj):
    adamopt.zero_grad()
    loss = obj(x)
    loss.backward()
    adamopt.step()
"""



#########################################################################################################


#Source: http://www.cmap.polytechnique.fr/~nikolaus.hansen/evco_11_1_1_0.pdf

def oldcmaes_step(x, obj, params):
    l = params["l"]
    mu = params["mu"]
    C = params["C"]
    pc = params["pc"]
    cc = params["cc"]
    ccov = params["ccov"]
    s = params["s"]
    ps = params["ps"]
    cs = params["cs"]
    ds = params["ds"]
    chi = params["chi"]
    n = x.size(0)
    
    #compute B and D
    B, D_squared, B_transpose = torch.linalg.svd(C)
    D = torch.diag(torch.sqrt(D_squared))
    BD = torch.matmul(B,D)
    
    #Sample offspring
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(n), torch.eye(n))
    z = m.sample((l,))
    offspring = x + s * torch.t(torch.matmul(BD, torch.t(z)))
        
    #Compute objective on each offspring
    evals = torch.zeros(l)
    for i in range(l):
        evals[i] = obj(offspring[i])
        
    #Get indices of top mu offspring
    top_evals, top_inds = torch.topk(evals, mu, largest=False)
    
    #Updates
    x_new = (1/mu) * torch.sum(offspring[top_inds], dim=0)
    z_avg = (1/mu) * torch.sum(z[top_inds], dim=0)
        
    #Covariance update
    pc_new = (1-cc)*pc + np.sqrt(cc*(2-cc)*mu) * torch.matmul(BD, z_avg)
    C_new = (1-ccov)*C + ccov*torch.outer(pc_new, pc_new)
    
    #Step size update
    ps_new = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu) * torch.matmul(B, z_avg)
    s_new = s * torch.exp((1/ds) * (torch.norm(ps_new)-chi) / chi)
    
    return x_new, C_new, pc_new, s_new, ps_new



def newcmaes_step(x, obj, params):
    l = params["l"]
    mu = params["mu"]
    C = params["C"]
    pc = params["pc"]
    cc = params["cc"]
    ccov = params["ccov"]
    acov = params["acov"]
    s = params["s"]
    ps = params["ps"]
    cs = params["cs"]
    ds = params["ds"]
    chi = params["chi"]
    n = x.size(0)
    
    #compute B and D
    B, D_squared, B_transpose = torch.linalg.svd(C)
    D = torch.diag(torch.sqrt(D_squared))
    BD = torch.matmul(B,D)
    
    #Sample offspring
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(n), torch.eye(n))
    z = m.sample((l,))
    offspring = x + s * torch.t(torch.matmul(BD, torch.t(z)))
        
    #Compute objective on each offspring
    evals = torch.zeros(l)
    for i in range(l):
        evals[i] = obj(offspring[i])
        
    #Get indices of top mu offspring
    top_evals, top_inds = torch.topk(evals, mu, largest=False)
    
    #Updates
    x_new = (1/mu) * torch.sum(offspring[top_inds], dim=0)
    z_avg = (1/mu) * torch.sum(z[top_inds], dim=0)
    
    #Covariance update
    pc_new = (1-cc)*pc + np.sqrt(cc*(2-cc)*mu) * torch.matmul(BD, z_avg)
    
    outer_prod_sum = torch.zeros(n,n)
    for i in range(mu):
        outer_prod_sum += (1/mu) * torch.outer(z[top_inds[i]], z[top_inds[i]])
    bigZ = torch.matmul(BD, torch.matmul(outer_prod_sum, torch.t(BD)))
    
    C_new = ((1-ccov)*C + 
             ccov * (acov*torch.matmul(pc_new, torch.t(pc_new)) +
                     (1-acov)*bigZ))
    
    #Step size update
    ps_new = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu) * torch.matmul(B, z_avg)
    s_new = s * torch.exp((1/ds) * (torch.norm(ps_new)-chi) / chi)
    
    return x_new, C_new, pc_new, s_new, ps_new


#Source: https://arxiv.org/pdf/1604.00772.pdf (see pg. 29)

def cmaes_step(x, obj, params):
    #Set parameters
    l = params["l"]
    mu = params["mu"]
    mu_eff = params["mu_eff"]
    C = params["C"]
    pc = params["pc"]
    cc = params["cc"]
    cm = params["cm"]
    c1 = params["c1"]
    cmu = params["cmu"]
    s = params["s"]
    ps = params["ps"]
    cs = params["cs"]
    ds = params["ds"]
    chi = params["chi"]
    w = params["w"]
    sumw = params["sumw"]
    epoch = params["epoch"]
    n = x.size(0)
    
    #Calculate B,D
    B, D_squared_diag, B_transpose = torch.linalg.svd(C)
    D_diag = torch.sqrt(D_squared_diag)
    D = torch.diag(D_diag)
    Dinv = torch.diag(1/D_diag)
    BD = torch.matmul(B,D)
    sqrtCinv = torch.matmul(B, torch.matmul(Dinv, B_transpose))
    
    #Sample offspring
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(n), torch.eye(n))
    z = m.sample((l,))
    y = torch.t(torch.matmul(BD, torch.t(z)))
    offspring = x + s*y
    
    #Compute objective on each offspring
    evals = torch.zeros(l)
    for i in range(l):
        evals[i] = obj(offspring[i])
    
    #Get indices of top mu offspring
    top_evals, top_inds = torch.topk(evals, l, largest=False)
    
    #Selection and recombination
    y_avg = torch.zeros(n)
    for i in range(mu):
        y_avg += w[i] * y[top_inds[i]]
    x_new = x + cm*s*y_avg
    
    #Step size update
    ps_new = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * torch.matmul(sqrtCinv, y_avg)
    s_new = s * torch.exp((cs/ds) * (torch.norm(ps)/chi - 1))

    #Covariance update    
    term1 = torch.norm(ps) / torch.sqrt(1 - (1-cs)**(2*(epoch+1)))
    term2 = (1.4 + 2/(n+1))*chi
    if term1 < term2:
        hs = 1
    else:
        hs = 0
    dhs = (1-hs) * cc * (2-cc)
    
    w_circle = torch.zeros(l)
    w_circle[:mu] = w[:mu]
    for i in range(mu, l):
        w_circle[i] = w[i] * n / torch.norm(torch.matmul(sqrtCinv, y[top_inds[i]]))**2
    
    pc_new = (1-cc)*pc + hs*np.sqrt(cc*(2-cc)*mu_eff)*y_avg

    outer_prod_sum = torch.zeros(n,n)
    for i in range(l):
        outer_prod_sum += w_circle[i] * torch.outer(y[top_inds[i]], y[top_inds[i]])
    
    C_new = ((1 + c1*dhs - c1 - cmu*sumw)*C +
         c1*torch.outer(pc, pc) +
         cmu*outer_prod_sum)
    
    return x_new, C_new, pc_new, s_new, ps_new