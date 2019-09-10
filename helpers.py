from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import pandas as pd
import fwd_sample as fs
import itertools

"""
Data cleaning helper functions
"""
def remove_cancellations(data):
    ## Guard against empty data
    if len(data) == 0:
        return -1, []

    cancel_count = 0
    cleaned_data = []
    for i, event in enumerate(data):
        ## Skip event and add to cancellation count if cancels
        if event[2] == -1.*data[i-1,2]:
            cancel_count += 1

        else:
            cleaned_data.append(event)

    return cancel_count, np.array(cleaned_data)

def remove_negatives(data):
    ## Guard against empty data
    if len(data) == 0:
        return -1, []

    neg_count = 0
    cleaned_data = []
    for i, event in enumerate(data):
        ## Skip event and add to negative count if negative
        if event[2] <= 0:
            neg_count += 1

        else:
            cleaned_data.append(event)

    return neg_count, np.array(cleaned_data)

def remove_outliers(data, amt_max, dur_max, rate_max, dur_min):
    ## Guard against empty data
    if len(data) == 0:
        return -1, []

    outlier_count = 0
    cleaned_data = []
    for i, event in enumerate(data):
        duration = (event[1] - event[0]).total_seconds()
        amt = event[2]
        rate = amt/duration

        if amt > amt_max:
            outlier_count += 1
            continue

        elif duration > dur_max:
            outlier_count += 1
            continue

        elif duration < dur_min:
            outlier_count += 1
            continue

        elif rate > rate_max:
            outlier_count += 1
            continue

        else:
            cleaned_data.append(event)

    return outlier_count, np.array(cleaned_data)

def clean_data(data, amt_max, dur_max, rate_max, dur_min):
    cancel_count, data = remove_cancellations(data)
    neg_count, data = remove_negatives(data)
    outlier_count, data = remove_outliers(data, amt_max, dur_max, rate_max, dur_min)
    return data, cancel_count, neg_count, outlier_count

def filter_data(df, cage_id, start, stop, max_cutoff):
    after_start = df[df['start_ts'] >= start].index
    correct_cage = df[df['cage_id'] == cage_id].index
    before_stop = df[df['end_ts'] <= stop].index
    within_cutoff = df[df['start_ts'] <= max_cutoff].index
    
    full_index = after_start.intersection(correct_cage).intersection(before_stop)
    
    if len(full_index) == 0:
        return [-1]
    
    next_after = pd.Index([full_index[-1] + 1])
    full_index = full_index.union(next_after)
    full_index = full_index.intersection(within_cutoff)
    full_index = full_index.intersection(correct_cage)
    
    return full_index

"""
Functions to read data files and extract empirical data
"""
def get_indiv(trace, idx):
    num_samples = trace.shape[0]
    data = trace[:, idx]

    return data

def get_indiv_theta(trace, idx):
    thetas = ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'theta7', 'theta8']
    post = []
    for theta in thetas:
        post.append(get_indiv(trace[theta], idx))

    return np.array(post)

def get_dataset(trace, rat_idx):
    thetas = ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'theta7', 'theta8']
    post = []
    for theta in thetas:
        post.append(trace[theta][:, rat_idx])

    return np.array(post)

def infer_duration(filename):
    drug, dose, recover, period, cage_id, duration, date = filename.split('_')
    return float(duration)

def rate_from_file(path, filename):
    dur = infer_duration(filename)
    filepath = path + '/' + filename
    data = np.loadtxt(filepath, delimiter='\t', usecols=(0,1,2,3,4))

    f_lengths = data[:,0]
    g_starts = data[:,1]
    rates = data[:,2]
    p_lengths = data[:,3]
    g_ends = data[:,4]
    
    qty = rates*f_lengths

    return sum(qty)/float(dur)

def amt_from_file(path, filename):
    filepath = path + '/' + filename
    data = np.loadtxt(filepath, delimiter='\t', usecols=(0,1,2,3,4))

    f_lengths = data[:,0]
    g_starts = data[:,1]
    rates = data[:,2]
    p_lengths = data[:,3]
    g_ends = data[:,4]
    
    qty = rates*f_lengths
    return sum(qty)

def amt_to_time(path, filename, maxtime):
    filepath = path + '/' + filename
    data = np.loadtxt(filepath, delimiter='\t', usecols=(0,1,2,3,4))

    t_elapsed = 0.
    food_intake = 0.
    i = 0
    while t_elapsed < maxtime and i < len(data):
        t_elapsed += data[i, 0]
        t_elapsed += data[i, 3]
        food_intake += data[i,0]*data[i,2]

        i += 1

    return food_intake

def cumulative_feeding(filename):
    data = np.loadtxt(filename, delimiter='\t', usecols=(0,1,2,3,4))
    
    ts = [0]
    for i in data:
        ## Append feeding data
        f_length = i[0]
        rate = i[2]
        
        for j in range(int(f_length)):
            ts.append(ts[-1]+rate*3.5)
            
        ## Append waiting data
        p_length = i[3]
        wait = int(p_length)*[ts[-1]]
        ts += wait
    
    return np.array(ts)

def group_c_feeding(folder):
    maxlen = 0
    x = []
    for filename in os.listdir(folder):
        new_ts = cumulative_feeding(folder + filename)
        if len(new_ts) > maxlen:
            maxlen = len(new_ts)
        x.append(np.array(new_ts))

    for i, ts in enumerate(x):
        x[i] = np.pad(ts, (0, maxlen-len(ts)), 'constant', constant_values=(0, np.nan))

    x = np.stack(x)
    return x

def group_amts(folder):
    amts = []
    for filename in os.listdir(folder):
        amts.append(amt_from_file(folder, filename))

    return amts

def get_filename(row):
    filename = '_'.join([row['drug'], 
                    str(row['dose']), 
                    row['adlib'],
                    row['period'],
                    str(row['id']),
                    str(row['duration']),
                    row['filename']])

    return filename


def data_from_row(row):
    filename = get_filename(row)

    filepath = row['filepath']

    data = np.loadtxt(filepath+'/'+filename, delimiter='\t', usecols=(0,1,2,3,4))
    return data

"""
Trace processing functions
"""
def run_name(tracename):
    return '_'.join(tracename.split('_')[:-1])


"""
Plotting helper functions
"""
def get_colour(data):
    drug, dose, recover, period = data
    dose = float(dose)
    #print drug

    if drug == 'PYY': # graded green
        if dose == 300:
            c = cm.tab20c(0.4)

        elif dose == 7.5:
            c = cm.tab20c(0.45)

        elif dose == 1.5:
            c = cm.tab20c(0.5)

        else:
            print 'ERROR with drug %s, dose %f' %(drug, dose)

    elif drug == 'LiCl': # graded orange
        if dose == 64:
            c = cm.tab20c(0.2)

        elif dose == 32:
            c = cm.tab20c(0.25)

        elif dose == 16:
            c = cm.tab20c(0.3)   
            
        else:
            print 'ERROR with drug %s, dose %f' %(drug, dose)         

    elif drug == 'GLP-1': # graded violet
        if dose == 300:
            c = cm.tab20c(0.6)

        elif dose == 100:
            c = cm.tab20c(0.65)

        elif dose == 30:
            c = cm.tab20c(0.7)   
            
        else:
            print 'ERROR with drug %s, dose %f' %(drug, dose)        

    elif drug == 'sib':
        #c = cm.tab20c(0.6) # yellow
        c= 'y'

    elif drug == 'Ex-4':
        #c = cm.tab20c(0.8) # grey
        c = 'k'

    elif drug == 'Lep':
        #c = cm.tab20c(0) # red
        c = 'r'

    elif drug == 'saline': # graded blue
        if recover == 'A':
            c = cm.tab20c(0.8) # grey

        elif recover == 'R':
            c = cm.tab20c(0) # blue

    elif drug == 'vehicle':
        c = cm.tab20c(0.05)

    else:
        print 'ERROR with drug %s' %(drug)

    return c


"""
Posterior helper functions
"""    
def get_indiv(trace, idx):
    num_samples = trace.shape[0]
    
    data_holder = []
    for i in range(0, num_samples):
        data_holder.append(trace[i, idx])
    
    data = np.stack(data_holder)
    return data

def cov_from_chol(num_vars, chol):
    cov = np.zeros((num_vars, num_vars))
    
    for i in range(0, num_vars):        
        chol_factor = chol[i*(i+1)/2:(i+1)*(i+2)/2]
        
        row_to_add = np.zeros(num_vars)
        row_to_add[0:len(chol_factor)] = chol[i*(i+1)/2:(i+1)*(i+2)/2]
        
        cov[i,:] = row_to_add
        
    cov = np.dot(cov, cov.T)
    
    return cov

def make_index(trace, varname):
    ## Get indices of individuals
    rat_idx = np.unique(trace[varname][0,:], return_index=True)[1]
    rat_idx = sorted(rat_idx)
    return rat_idx

def get_posterior(trace, idx, varlist):
    ## Get posterior mean values    
    theta_holder = []
    for var in varlist:
        theta_holder.append(get_indiv(trace[var], idx))

    theta_holder = np.array(theta_holder).astype(float)
    return theta_holder

def make_single_pm_df(trace, filenames, varlist):
    ## Get indices of individuals
    rat_idx = make_index(trace, varlist[0])

    ## Get posterior mean values
    theta_holder = get_posterior(trace, rat_idx, varlist)
    theta_holder = np.mean(theta_holder, axis=1).T

    ## Get rat info
    data_holder = []
    for i, filename in enumerate(filenames):
        data = filename.split('_')
        filepath = 'new_all_data/'+'_'.join(data[:4])
        amt = amt_from_file(filepath, filename)
        rate = amt/float(data[5]) # row[5] is duration
        #print filepath + '/' + filename
        x0 = np.loadtxt(filepath+'/'+filename, delimiter='\t', usecols=(0,1,2,3,4))[0][1] # g_start
        data = data + [filepath, rate, x0]
        data_holder.append(data)

    ## Create the dataframe
    datalist = ['drug', 'dose', 'adlib', 'period', 'id', 'duration', 'filename', 'filepath', 'rate', 'x0']
    pre_df = np.concatenate([theta_holder, data_holder], axis=1)
    columns = varlist + datalist
    df = pd.DataFrame(pre_df, columns=columns)

    return df

def make_pm_dataframe(trace, subjs, varlist, datalist):
    ## Get indices of individuals
    rat_idx = make_index(trace, varlist[0])

    ## Get posterior mean values    
    theta_holder = get_posterior(trace, rat_idx, varlist)
    theta_holder = np.mean(theta_holder, axis=1).T

    ## Get rat information
    data_holder = []
    for i, subj in enumerate(subjs):
        filepath = '/'.join(subj)
        data = subj[1].split('_')
        data[-1] = subj[1]
        data = data + [subj[0]]
        data_holder.append(data)

    data_holder = np.array(data_holder)

    ## Join the two into a dataframe
    pre_df = np.concatenate([theta_holder, data_holder], axis=1)
    columns = varlist + datalist
    df = pd.DataFrame(pre_df, columns=columns)

    ## Get feeding amount
    def get_rate(row):
        amt = amt_from_file(row['filepath'], row['filename'])
        return amt/float(row['duration'])

    df['rate'] = df.apply(get_rate, axis=1)

    return df

def add_post_sample_dict(data_dict, trace, subjs, varlist, num_samples=100):
    ## Get indices of individuals
    rat_idx = make_index(trace, varlist[0])

    ## Get posterior mean values    
    theta_holder = get_posterior(trace, rat_idx, varlist)
    post_size = theta_holder.shape[1]

    ## Sample from posterior
    sample_idx = np.random.randint(0, post_size, num_samples)
    samples = theta_holder[:, sample_idx, :]

    ## Get rat information
    for i, subj in enumerate(subjs):
        data_dict[subj] = samples[:,:,i]

    return data_dict

def add_group_post_dict(data_dict, trace, filename, num_samples=100):
    ## Extract group mean posterior only
    group_means = trace['mu']
    post_size = group_means.shape[0]

    ## Sample from posterior
    sample_idx = np.random.randint(0, post_size, num_samples)
    samples = group_means[sample_idx, :]

    ## Store in the dict
    data_dict[filename.split('/')[1]] = samples

    return data_dict


"""
def make_posterior_dict(trace, subjs, paths, varlist):
    ## Get indices of individuals
    rat_idx = make_index(trace, varlist[0])

    ## Get full posterior   
    theta_holder = get_posterior(trace, rat_idx, varlist)

    ## Create dictionary
    post_dict = {}
    for i, subj in enumerate(subjs):
        ## Filepath will be dictionary key
        filepath = '/'.join(subj)

        ## Store the metadata (including amount eaten) in the dictionary
        data = subj[1].split('_')
        data[-1] = subj[1]
        data = data + [subj[0]]
        amt = amt_from_file(subj[0], subj[1])
        data = [amt] + data

        post_dict[filepath] = [data, theta_holder[:,:,i]]

    return post_dict
"""
# def sample_group(trace, group_id, sample_num):
#     mu = trace['mu'][sample_num,group_id,:]
    
#     chol = trace['chol_cov'+str(group_id)][sample_num,:] # modified to include different cov matrices
#     cov = cov_from_chol(8, chol)
    
#     return mu, cov

def group_hist(trace, group_list, group_id_dict, theta_idx, recip=False):
    fig, axes = plt.subplots(2,1)

    theta = trace['mu'][:,:,theta_idx]

    for i in group_list:
        theta_var = np.power(10., theta[:,i])
        
        if recip == True:
            theta_var = 1./theta_var

        data = group_id_dict[i].split('_')
        c = get_colour(data)
        
        if data[3] == 'D':
            axes[0].hist(theta_var, label=group_id_dict[i],
                         color=c, alpha=0.6, bins=20, normed=True)
            
        elif data[3] == 'L':
            axes[1].hist(theta_var, label=group_id_dict[i],
                         color=c, alpha=0.6, bins=20, normed=True)  
            
        else:
            raise ValueError
            
    axes[0].legend()
    return fig, axes


"""
Model functions
"""

"""
Forward sampling helper functions
"""
def make_protocol_list(druglist, protocol_size, duration, min_default):
    default_drug = druglist[0]
    protocol_list = itertools.combinations_with_replacement(druglist, protocol_size)
    
    ## Get all unique protocols
    protocols = []
    for i in protocol_list:
        perms = itertools.permutations(i)
        for j in perms:
            if j not in protocols:
                protocols.append(j)

    ## Add duration information
    protocols_with_durations = []
    for protocol in protocols:
        default_count = 0
        new_protocol = []
        for part in protocol:
            new_protocol.append((duration, part))
            if part == default_drug:
                default_count += 1

        if default_count >= min_default:
            protocols_with_durations.append(new_protocol)

        else: continue

    return protocols_with_durations

def sample_protocol(data_dict, protocol, num_samples, cutoff, pc=5):
    ## NOTE: currently does not switch posterior through a long pause
    ## This should probably be fixed
    max_duration = 3600*sum([i[0] for i in protocol])

    ## Repeatedly sample from the posteriors sequentially
    all_ts = []
    amounts = []
    for i in range(0, num_samples):
        x0 = 0
        total_amount = 0
        time_series = [[x0]]
        final_state = 'F'
        
        for state in protocol:
            duration = state[0]*3600
            post = np.mean(data_dict[state[1]], axis=0)
            x0 = time_series[-1][-1]
            results = fs.sample(duration, post, x0, init_state=final_state)
            
            time_series.append(results[0][:duration])
            total_amount += results[1]
            
            ## Use cutoff to check pause state
            events = results[-1]
            if events[-1][3] > cutoff: # would need to redo sampling to get last state otherwise
                final_state = 'L'

            else:
                final_state = 'S'

        time_series = np.hstack(time_series)
        all_ts.append(time_series[:max_duration])
        amounts.append(total_amount)

    all_ts = np.array(all_ts)
    last_ts = time_series[:max_duration]

    ts_mean = np.mean(all_ts, axis=0)
    ts_pc_low = np.percentile(all_ts, pc, axis=0)
    ts_pc_high = np.percentile(all_ts, 100-pc, axis=0)
    ts_pc = (ts_pc_low, ts_pc_high)

    return amounts, ts_mean, ts_pc, last_ts

def sample_group(mu, cov, groupsize, duration):
    thetas = np.random.multivariate_normal(mu, cov, size=groupsize)

    amts = []
    for i in range(groupsize):
        t_start = datetime.now()
        results = fs.sample(duration, thetas[i,:], 0)
        t_end = datetime.now()
        amts.append(results[1])

    return amts