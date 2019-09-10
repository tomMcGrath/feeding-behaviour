import sys
import os
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import PDMP_ll as ll
import cPickle as pickle

datapath = sys.argv[1]
num_tune = int(sys.argv[2])
num_samples = int(sys.argv[3])

run_name = datapath.split('/')[1]

"""
Data import
"""
min_num_animals = 5

def data_from_file(filename):
    data = np.loadtxt(filename,delimiter='\t',usecols=(0,1,2,3,4))

    f_lengths = data[:,0]
    g_starts = data[:,1]
    rates = data[:,2]
    p_lengths = data[:,3]
    g_ends = data[:,4]

    obs = [f_lengths, g_starts, rates, p_lengths, g_ends]
    for i in p_lengths:
        if i == 0:
            print i
    return obs

filenames = os.listdir(datapath)

idx_holder = []
data_holder = []

for i, filename in enumerate(filenames):
    data = data_from_file(datapath + '/' + filename)
    data_holder.append(data)
    idx_holder.append(i*np.ones(len(data[0])))

data = np.hstack(data_holder)
idx = np.hstack(idx_holder).astype(int)
group_size = len(filenames)

"""
Model code
"""
## Set constants
k1 = 0.00055

## Model setup
with pm.Model() as model:
    ## Group mean
    means = [-3, -3, -3, 1, 1, -1, 3, 3] # from unpooled data, all vars
    num_vars = len(means)
    cov = np.eye(num_vars)

    mu = pm.Normal('mu', mu=means, sd=3, shape=num_vars)

    theta_holder = []

    ## Create covariance matrix from LKJ
    sd_dist = pm.HalfNormal.dist(sd=3, shape=num_vars)
    packed_chol = pm.LKJCholeskyCov('chol_cov', eta=1, n=num_vars, sd_dist=sd_dist)
    chol = pm.expand_packed_triangular(num_vars, packed_chol, lower=True)

    theta_tilde = pm.Normal('theta_tilde', mu=0, sd=1, shape=(group_size, num_vars))
    theta_gp = tt.dot(chol, theta_tilde.T).T # have to take transpose to fit with distribution shape
    theta_holder.append(theta_gp)

    thetas = tt.concatenate(theta_holder)

    theta1 = pm.Deterministic('theta1', mu[0] + thetas[idx, 0]) 
    theta2 = pm.Deterministic('theta2', mu[1] + thetas[idx, 1]) 
    theta3 = pm.Deterministic('theta3', mu[2] + thetas[idx, 2]) 
    theta4 = pm.Deterministic('theta4', mu[3] + thetas[idx, 3]) 
    theta5 = pm.Deterministic('theta5', mu[4] + thetas[idx, 4]) 
    theta6 = pm.Deterministic('theta6', mu[5] + thetas[idx, 5]) 
    theta7 = pm.Deterministic('theta7', mu[6] + thetas[idx, 6]) 
    theta8 = pm.Deterministic('theta8', mu[7] + thetas[idx, 7])
    
    """
    Power-transform
    """
    p10_theta1 = tt.pow(10., theta1)
    p10_theta2 = tt.pow(10., theta2)
    p10_theta3 = tt.pow(10., theta3)
    p10_theta6 = tt.pow(10., theta6)
    p10_theta7 = tt.pow(10., theta7)
    p10_theta8 = tt.pow(10., theta8)

    ## Likelihood of observations

    ## Exponential feeding bout length
    feeding_lengths = pm.Exponential('f_len', p10_theta1, observed=data[0,:])

    ## Normal feeding bout rate
    rates = pm.Normal('rate', p10_theta2, sd=p10_theta3, observed=data[2,:])

    ## Pause likelihood
    pauses = ll.pause_ll('pause', theta4, theta5, p10_theta6, p10_theta7, p10_theta8, k1, observed=data)

    ## Checking out different step methods to see which works
    # NUTS w/o ADVI - currently fails on LKJCholeskyCov
    
    trace = pm.sample(num_samples, 
                      tune=num_tune, 
                      njobs=1, 
                      step=pm.NUTS(), 
                      target_accept=0.7, 
                      max_treedepth=10)

    pm.traceplot(trace)
    plt.savefig(run_name+'_traceplot.png')

"""
Export the data
"""
    
pickle.dump(trace, open(run_name+"_trace.p", "wb"))
pickle.dump(idx, open(run_name+"_idx.p", "wb"))
pickle.dump(filenames, open(run_name+"_filenames.p", "wb"))
