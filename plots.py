import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import helpers
import os
import pandas as pd
import scipy
import scipy.stats
import fwd_sample as fs
import fwd_likelihoods as fl
import scipy
import itertools
import datetime

"""
Figure prelims
"""
def compare_sample(df, data_dict, varlist, idx):
	fig, axes = plt.subplots(1, figsize=(5,5))
	var = varlist[idx]
	
	def compare(row):
		x = row[var]
		filename = helpers.get_filename(row)
		post = data_dict[filename]
		y = np.mean(post, axis=1)[idx]
		axes.scatter(x, x-y)

	df.apply(compare, axis=1)

	return fig, axes

"""
Figure 1
"""
def ts_from_data(data):
	k1 = 0.00055
	bout_ts_holder = [[0]]
	stomach_ts_holder = [[0]]

	## Generate the base time series
	for event in data:
		f_length, g_start, rate, p_length, g_end_feeding = event

		## Bout ts holder gets one entry at rate for time f_length
		## and one entry at rate 0 for time p_length
		bout_ts_holder.append(rate*np.ones(int(f_length)))
		bout_ts_holder.append(np.zeros(int(p_length)))

		## Stomach ts holder gets a linearly increasing line at rate for f_length
		## then decreases according to the ODE
		g_start = stomach_ts_holder[-1][-1]
		stomach_ts_holder.append(g_start + 3.5*rate*(1+np.arange(int(f_length)))) # feeding

		g_end = stomach_ts_holder[-1][-1]
		t_c = 2.*np.sqrt(g_end)/k1		
		
		if p_length <= t_c:
			digestion_ts = 0.25*np.power((2.*np.sqrt(g_end) - k1*(1+np.arange(int(p_length)))), 2)
			stomach_ts_holder.append(digestion_ts)

		else:
			digestion_ts1 = 0.25*np.power((2.*np.sqrt(g_end) - k1*(1+np.arange(int(t_c)))), 2)
			digestion_ts2 = np.zeros(int(p_length - t_c))
			stomach_ts_holder.append(digestion_ts1)
			stomach_ts_holder.append(digestion_ts2)

	## Create a time series from the time series holder
	bout_ts = np.hstack(bout_ts_holder)
	stomach_ts = np.hstack(stomach_ts_holder)

	return bout_ts, stomach_ts

def timeseries_predict_plot(data, thetas, predict_index, num_samples=100, tmax=60*60):
	k1 = 0.00055
	theta7 = np.power(10., thetas[6])
	theta8 = np.power(10., thetas[7])
	fig, axes = plt.subplots(2,1, figsize=(10,5))

	bout_ts, stomach_ts = ts_from_data(data)
	x_bout_sec = np.arange(len(bout_ts))
	x_stom_sec = np.arange(len(stomach_ts))

	x_bout_min = x_bout_sec/60
	x_stom_min = x_stom_sec/60

	print len(x_bout_sec), len(bout_ts), len(x_bout_min)
	print len(x_bout_sec), len(bout_ts), len(x_bout_min)

	## Plot the time series
	axes[0].plot(x_bout_sec/60., bout_ts)
	axes[1].plot(x_stom_sec/60., stomach_ts)

	#ax2 = axes[1].twinx()
	
	## Now do the posterior prediction of next mealtime
	## Get the time of the start of the intermeal interval we wish to predict
	known_events = data[:predict_index]
	start_time = 0
	for event in known_events[:-1]: # only want feeding from the last one
		start_time += event[0] + event[3]

	start_time += known_events[-1][0] # add the g_end_feeding
	x0 = stomach_ts[int(start_time)]

	next_times = []
	i = 0
	while i < num_samples:
		new_sample = fl.sample_L(x0, k1, theta7, theta8)
		next_times.append(start_time + new_sample)

		i += 1

	axes[0].set_ylim(0, 0.02)
	ax2 = axes[0].twinx()

	## Histogram of results
	#ax2.hist(next_times, histtype='step', color='r', bins=100, normed=True)

	## KDE of results
	x_grid = np.arange(len(stomach_ts))
	kde = scipy.stats.gaussian_kde(next_times, bw_method='silverman')
	y = kde.evaluate(x_grid)
	ax2.plot(x_grid[int(start_time):]/60., y[int(start_time):], c='r')
	ax2.fill_between(x_grid[int(start_time):30000]/60., 0, y[int(start_time):30000], color='r', alpha=0.3)
	ax2.set_ylim([0, 2.5*np.max(y)])
	ax2.set_yticklabels([])
	
	## Predict samples of stomach fullness
	ts_predictions = []
	i = 0
	while i < num_samples:
		new_sample = fs.sample(tmax, thetas, x0, init_state='L')
		ts_predictions.append(new_sample[0][:tmax])
		i += 1

	ts_predictions = np.stack(ts_predictions)
	x = start_time + np.arange(tmax) # offset to start time

	## Plot mean
	mean_ts = np.mean(ts_predictions, axis=0)
	axes[1].plot(x/60., mean_ts, c='r')
	
	## Plot percentile
	"""
	pc = 5
	min_val = np.percentile(ts_predictions, pc, axis=0)
	max_val = np.percentile(ts_predictions, 100-pc, axis=0)
	axes[1].fill_between(x, min_val, max_val, alpha=0.3, color='r')
	"""
	
	# ## Plot samples
	for i in range(5):
		axes[1].plot(x/60., ts_predictions[i, :], c='r', alpha=0.3)

	# plt.subplots_adjust(hspace=0.1)
	
	return fig, axes, ax2

def timeseries_inset(data, predict_index):
	k1 = 0.00055
	fig, axes = plt.subplots(1,1, figsize=(3,3))

	bout_ts, stomach_ts = ts_from_data(data)
	x_bout_sec = np.arange(len(bout_ts))
	x_stom_sec = np.arange(len(stomach_ts))

	x_bout_min = x_bout_sec/60
	x_stom_min = x_stom_sec/60

	print len(x_bout_sec), len(bout_ts), len(x_bout_min)
	print len(x_bout_sec), len(bout_ts), len(x_bout_min)

	## Plot the time series
	axes.plot(x_bout_sec/60., bout_ts)
	
	return fig, axes

def plot_ethogram(folder, num_animals, maxlen=None, downsample=10, ysize=50):
	num_animals = min(num_animals, len(os.listdir(folder)))

	fig, axes = plt.subplots(2*num_animals, 1, figsize=(5,5))

	for i, filename in enumerate(os.listdir(folder)):
		if i >= num_animals:
			continue

		filepath = folder + '/' + filename
		data = np.loadtxt(filepath, delimiter='\t', usecols=(0,1,2,3,4))
		bout_ts, stomach_ts = ts_from_data(data)

		## Set the feeding to binary and downsample
		bout_ts = bout_ts[:maxlen] > 0.0
		bout_ts = bout_ts[::downsample]

		## Downsample the stomach fullness
		stomach_ts = stomach_ts[:maxlen]
		stomach_ts = stomach_ts[::downsample]

		## Make rasterplot of bouts
		x = np.outer(np.ones(ysize), bout_ts)
		axes[2*i].matshow(x, cmap=cm.Greys)
		## Plot stomach fullness
		y = np.outer(np.ones(ysize), stomach_ts)
		axes[2*i + 1].matshow(y, cmap=cm.YlOrRd)

	for i in range(2*num_animals):
		axes[i].set_xticklabels([])
		axes[i].set_yticklabels([])
		axes[i].set_yticks([])

	#fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.001)
	plt.subplots_adjust(hspace=0.05)
	return fig, axes


"""
Figure 2
"""
def pairplot(df, var1, var2, ctype='drug_c', transforms=[None, None], figsize=(5,5)):
	fig, axes = plt.subplots(1, figsize=figsize)

	def plot_pairs(row):
		x = row[var1]
		y = row[var2]
		c = row[ctype]
		ms = row['ms']

		if transforms[0] == '-1':
			x = -1*x

		if transforms[1] == '-1':
			y = -1*y

		if transforms[0] == 'pow10':
			x = np.power(10., x)

		if transforms[1] == 'pow10':
			y = np.power(10., y)

		if transforms[0] == 'pow10_inv':
			x = np.power(10., -x)

		if transforms[1] == 'pow10_inv':
			y = np.power(10., -y)

		axes.scatter(x, y, c=c, marker=ms)

	df.apply(plot_pairs, axis=1)

	return fig, axes

def trellisplot(df, varlist, transforms):
	## Prepare the figure
	num_vars = len(varlist)
	fig, axes = plt.subplots(num_vars, num_vars, figsize=(10,10))

	## Plot the data
	for i, var1 in enumerate(varlist):
		dataset1 = df[var1]
		transform1 = transforms[i]
		if transform1 == 'pow10_inv':
			dataset1 = np.power(10., -dataset1)

		for j, var2 in enumerate(varlist):
			dataset2 = df[var2]
			transform2 = transforms[j]
			if transform2 == 'pow10_inv':
				dataset2 = np.power(10., -dataset2)

			## Data on the lower diagonal

			## KDE/histogram on the diagonal
			if i == j:
				axes[i,i].hist(dataset1, bins=20, normed=True)

			else:
				axes[j, i].scatter(dataset1, dataset2, alpha=0.6)



	## Do regression - could bootstrap
	for i, var1 in enumerate(varlist):
		dataset1 = df[var1]
		transform1 = transforms[i]
		if transform1 == 'pow10_inv':
			dataset1 = np.power(10., -dataset1)

		for j, var2 in enumerate(varlist):
			dataset2 = df[var2]
			transform2 = transforms[j]
			if transform2 == 'pow10_inv':
				dataset2 = np.power(10., -dataset2)

			if i == j:
				continue

			else:
				## Do the regression
				x = dataset1
				y = dataset2
				slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)

				axes[j, i].plot(x, intercept + x*slope, c='k')
				#text = '$r^{2} = %0.2f$\n$p=%1.0e$' %(r_value**2, p_value)
				#axes[i, j].text(0.15, 0.4, text, fontsize=8)

	return fig, axes

def univariate_posterior(data_dict, idx, groups_to_use, numbins=50, transform=None):
	fig, axes = plt.subplots(2, 1, figsize=(4,5))

	for key in data_dict.keys():
		if key not in groups_to_use:
			continue

		dataset = data_dict[key][:, idx]
		data = key.split('_')[:-1]
		c = helpers.get_colour(data)

		if transform == 'pow10':
			dataset = np.power(10., dataset)

		if transform == 'pow10_inv':
			dataset = np.power(10., -dataset)

		if data[3] == 'D':
			axes[0].hist(dataset, bins=numbins, color=c, histtype='stepfilled', alpha=0.6, normed=True)

		else:
			axes[1].hist(dataset, bins=numbins, color=c, histtype='stepfilled', alpha=0.6, normed=True)

	#axes[0].set_yticklabels([])
	#axes[1].set_yticklabels([])

	return fig, axes

"""
Figure 3
"""
def IMI_curve(group_dict, num_points=10, num_resamples=10):
	fig, axes = plt.subplots(1, figsize=(5,5))
	k1 = 0.00055

	x0_vals = np.linspace(0, 20, num_points)

	for key in group_dict.keys():
		group_post = group_dict[key]
		data = key.split('_')[:-1]
		c = helpers.get_colour(data)

		predicted_IMIs = []
		for x0 in x0_vals:
			this_x0_predictions = []
			for sample in group_post:
				theta7 = np.power(10., sample[6])
				theta8 = np.power(10., sample[7])

				for repeat in range(num_resamples):
					this_x0_predictions.append(fl.sample_L(x0, k1, theta7, theta8))

			predicted_IMIs.append(np.mean(this_x0_predictions))

		axes.plot(x0_vals, predicted_IMIs, c=c)

	return fig, axes

def IMI_fullness(df, cutoff=300):
	fig, axes = plt.subplots(1, figsize=(5,5))
	k1 = 0.00055

	def fullness_IMI(row):
		## Import the meal data for this animal
		data = helpers.data_from_row(row)

		## Iterate through the bouts, getting x0 and IMI once pause length exceeds cutoff
		true_IMIs = []
		g_ends = [] # predictions based on individual posterior means

		for event in data:
			f_length, g_start, rate, p_length, g_end_feeding = event

			IMI_samples = []
			if p_length > cutoff:
				g_ends.append(g_end_feeding)
				true_IMIs.append(p_length)

		## Plot the results
		axes.scatter(g_ends, true_IMIs, c=row['rate_c'], alpha=0.3)

	## Iterate over the dataset
	df.apply(fullness_IMI, axis=1)

	## Axes labels etc
	axes.set_xlabel('Stomach fullness')
	axes.set_ylabel('Observed IMI')

	axes.set_yscale('log')

	return fig, axes

def IMI_prediction(df, num_samples=10, cutoff=300):
	fig, axes = plt.subplots(3, 1, figsize=(5,5))
	k1 = 0.00055

	def predict_IMI(row):
		## Import the meal data for this animal
		data = helpers.data_from_row(row)

		## Get PM thetas
		theta7 = np.power(10., row['theta7'])
		theta8 = np.power(10., row['theta8'])

		## Iterate through the bouts, getting x0 and IMI once pause length exceeds cutoff
		true_IMIs = []
		indiv_predicts = [] # predictions based on individual posterior means
		residuals = []
		next_sizes = []
		for i, event in enumerate(data):
			f_length, g_start, rate, p_length, g_end_feeding = event

			IMI_samples = []
			if p_length > cutoff:
				x0 = g_end_feeding
				true_IMIs.append(p_length)

				for j in range(0, num_samples):
					IMI_samples.append(fl.sample_L(x0, k1, theta7, theta8))

				mean_predict = np.mean(IMI_samples)
				indiv_predicts.append(mean_predict)
				resid = p_length - mean_predict
				residuals.append(resid)

				if i < len(data)-1:
					next_size = 3.5*data[i+1][0]*data[i+1][2]
					next_sizes.append(next_size)

		## Plot the results
		#axes.scatter(true_IMIs, indiv_predicts, c=row['drug_c'], alpha=0.1)
		#axes.scatter(true_IMIs, indiv_predicts, c=row['rate_c'], alpha=0.1)
		axes[0].scatter(true_IMIs, indiv_predicts, c='b', alpha=0.1)
		axes[1].scatter(true_IMIs, residuals, c='b', alpha=0.1)
		axes[2].scatter(residuals[:len(next_sizes)], next_sizes, c='b', alpha=0.025)

	## Iterate over the dataset
	df.apply(predict_IMI, axis=1)

	## Axes labels etc
	axes[0].set_xlabel('True IMI')
	axes[0].set_ylabel('Predicted IMI')

	#axes.set_xlim([0, 20000])
	#axes.set_ylim([0, 20000])

	## Visual guides
	x = np.linspace(cutoff, 8*3600, 10)
	axes[0].plot(x, x, c='k', ls='--')
	axes[1].axhline(0, c='k', ls='--')

	axes[0].set_xscale('log')
	axes[0].set_yscale('log')
	axes[1].set_xscale('log')

	return fig, axes

def predict_IMI_full_post(df, data_dict, num_resamples=1, cutoff=300):
	fig, axes = plt.subplots(1, figsize=(5,5))
	k1 = 0.00055

	def predict_IMI(row):
		## Import the meal data for this animal
		data = helpers.data_from_row(row)

		## Iterate through the bouts, getting x0 and IMI once pause length exceeds cutoff
		true_IMIs = []
		indiv_predicts = [] # predictions based on individual posterior means
		indiv_errs = []

		## Load the posterior sample dictionary
		filename = helpers.get_filename(row)
		post = data_dict[filename]

		for event in data:
			f_length, g_start, rate, p_length, g_end_feeding = event

			IMI_samples = []
			if p_length > cutoff:
				x0 = g_end_feeding
				true_IMIs.append(p_length)

				for i in range(post.shape[1]):
					post_sample = post[:,i]
					theta7 = np.power(10., post_sample[6])
					theta8 = np.power(10., post_sample[7])
					IMI_samples.append(fl.sample_L(x0, k1, theta7, theta8))

				indiv_predicts.append(np.mean(IMI_samples))
				indiv_errs.append(np.std(IMI_samples))

		## Plot the results
		axes.scatter(true_IMIs, indiv_predicts, c=row['drug_c'], marker=row['ms'])
		"""
		axes.errorbar(true_IMIs, 
					  indiv_predicts, 
					  yerr=indiv_errs, 
					  c=row['drug_c'], 
					  fmt='o',
					  marker=row['ms'])
		"""
	## Iterate over the dataset
	df.apply(predict_IMI, axis=1)

	## Axes labels etc
	axes.set_xlabel('True IMI')
	axes.set_ylabel('Predicted IMI')

	#axes.set_xlim([0, 20000])
	#axes.set_ylim([0, 20000])

	x = np.linspace(cutoff, 8*3600, 10)
	axes.plot(x, x, c='k', ls='--')

	axes.set_xscale('log')
	axes.set_yscale('log')

	return fig, axes

def IMI_prediction_KDEmax(df, num_samples=10, cutoff=300):
	fig, axes = plt.subplots(1, figsize=(5,5))
	k1 = 0.00055

	def predict_IMI(row):
		## Import the meal data for this animal
		data = helpers.data_from_row(row)

		## Get PM thetas
		theta7 = np.power(10., row['theta7'])
		theta8 = np.power(10., row['theta8'])

		## Iterate through the bouts, getting x0 and IMI once pause length exceeds cutoff
		true_IMIs = []
		indiv_predicts = [] # predictions based on individual posterior means

		for event in data:
			f_length, g_start, rate, p_length, g_end_feeding = event

			IMI_samples = []
			if p_length > cutoff:
				x0 = g_end_feeding
				true_IMIs.append(p_length)

				for i in range(0, num_samples):
					IMI_samples.append(fl.sample_L(x0, k1, theta7, theta8))

				## Make a KDE of the samples and find the maximum value
				x_grid = np.linspace(0, max(IMI_samples), 10000)
				kde = scipy.stats.gaussian_kde(IMI_samples)
				kde_vals = kde.evaluate(x_grid)
				KDEmax = np.argmax(kde_vals)
				#print x_grid[KDEmax]
				indiv_predicts.append(x_grid[KDEmax])

		## Plot the results
		#axes.scatter(true_IMIs, indiv_predicts, c=row['drug_c'], alpha=0.1)
		#axes.scatter(true_IMIs, indiv_predicts, c=row['rate_c'], alpha=0.1)
		axes.scatter(true_IMIs, indiv_predicts, c='b', alpha=0.1)

	## Iterate over the dataset
	df.apply(predict_IMI, axis=1)

	## Axes labels etc
	axes.set_xlabel('True IMI')
	axes.set_ylabel('Predicted IMI')

	#axes.set_xlim([0, 5000])
	#axes.set_ylim([0, 5000])

	x = np.linspace(cutoff, 8*3600, 10)
	axes.plot(x, x, c='k', ls='--')

	axes.set_xscale('log')
	axes.set_yscale('log')

	return fig, axes

def intake_fullness(df, cutoff=300):
	fig, axes = plt.subplots(1, figsize=(5,5))

	def plot_meals(row):
		## Import the meal data for this animal
		data = helpers.data_from_row(row)
		
		## Iterate through the bouts, storing meal data once pause length exceeds cutoff
		mealsizes = []
		gut_ends = []
		this_mealsize = 0
		for event in data:
			f_length, g_start, rate, p_length, g_end_feeding = event

			if p_length > cutoff:
				this_mealsize += 3.5*rate*f_length # add the feeding in this meal
				mealsizes.append(this_mealsize) # store
				this_mealsize = 0 # reset for next meal
				gut_ends.append(g_end_feeding) # store g_end

			else:
				this_mealsize += 3.5*rate*f_length

		c = [row['rate_c']]*len(data) # matplotlib expects a vector of colours - if there are 4 points it breaks!
		ms = row['ms']

		axes.scatter(gut_ends, mealsizes, c=c, marker=ms)

	## Apply the plot function across the dataset
	df.apply(plot_meals, axis=1)

	## Set labels etc
	axes.set_xlabel('Stomach fullness at meal termination (kcal)')
	axes.set_ylabel('Meal size (kcal)')


	return fig, axes

def fullness_IMI(df, cutoff=300, exp_param=1.5):
	fig, axes = plt.subplots(2, 1, figsize=(5,5))

	p_lengths = []
	def plot_data(row):
		## Import the meal data for this animal
		data = helpers.data_from_row(row)

		for event in data:
			f_length, g_start, rate, p_length, g_end_feeding = event
			
			#p_lengths.append(np.log10(p_length))
			if p_length < cutoff:
				p_lengths.append(p_length)

			axes[0].scatter(g_end_feeding, p_length, c='b', alpha=0.1)

	df.apply(plot_data, axis=1)

	lambd = np.power(10., exp_param)
	samples = np.random.exponential(lambd, size=1000)
	#samples = np.log10(samples)

	axes[0].axhline(cutoff, c='k', ls='--')
	axes[0].set_yscale('log')
	axes[1].hist([samples, p_lengths], bins=10, normed=True, color=['b', 'r'])
	#axes[1].hist(p_lengths, bins=20, normed=True)
	#axes[1].axvline(np.log10(cutoff), c='k', ls='--')
	axes[1].axvline(cutoff, c='k', ls='--')

	return fig, axes

def plot_IMI(data_dir, data_dict, cutoff=300, num_samples=10, windowsize=2):
    err_thresh = 20000
    
    p_lengths = []
    g_ends = []
    c = []

    ## Assemble dataset
    count = len(os.listdir(data_dir))
    for i, dataset in enumerate(os.listdir(data_dir)):
        data = np.loadtxt(data_dir+dataset, delimiter='\t', usecols=(0,1,2,3,4))

        for j in data:
            f_lengths, g_starts, rates, p_length, g_end = j

            if p_length > cutoff and p_length < err_thresh:
                p_lengths.append(p_length/60) # convert to minutes
                g_ends.append(g_end)
                c.append(float(i)/count)

            if p_length > err_thresh:
                print 'Error in %s, IMI of %3.0f' %(dataset, p_length)

    ## Process to use in moving window
    results = zip(g_ends, p_lengths)
    results = np.array(results)

    ## Generate moving window mean
    x_grid = np.linspace(0, np.max(results[:,0]), 100)

    means = []
    for xval in x_grid:
        usevals = np.abs(results[:,0] - xval) < windowsize
        meanval = np.mean(results[usevals, 1])
        means.append(meanval)

    """
    for i in results:
        print '%3.3f\t%3.3f' %(i[0], i[1])
    """
    means = np.array(means)

    ## Plot moving window
    fig, axes = plt.subplots(1, figsize=(5,5))
    axes.scatter(results[:,0], 
                 results[:,1], 
                 alpha=0.3, 
                 c='b', # use c=c to get individual colour coding
                 cmap=cm.gist_ncar)
    
    axes.plot(x_grid, means, c='k')
    
    ## Generate posterior predictive curve
    post = data_dict[data_dir.split('/')[1]+'_trace.p']
    post = np.mean(post, axis=0)
    
    mean_ppcs = []
    ppcs_low = []
    ppcs_high = []
    k1 = 0.00055
    sample_x_grid = np.linspace(0, np.max(results[:,0]), 30)
    for xval in sample_x_grid:
        samples = []
        for i in range(num_samples):
            theta7 = np.power(10., post[6])
            theta8 = np.power(10., post[7])
            samples.append(fl.sample_L(xval, k1, theta7, theta8)/60) # convert to minutes
            
        mean_ppcs.append(np.mean(samples))
        ppc_low = np.percentile(samples, 5)
        ppc_high = np.percentile(samples, 95)
        ppcs_low.append(ppc_low)
        ppcs_high.append(ppc_high)
        
    axes.plot(sample_x_grid, mean_ppcs, c='r')
    axes.fill_between(sample_x_grid, ppcs_low, ppcs_high, color='r', alpha=0.3)

    return fig, axes

def plot_satiety_ratio(data_dir, cutoff=300, windowsize=2):
    p_lengths = []
    mealsizes = []
    c = []
    err_thresh = 20000
    ratios = []

    ## Assemble dataset
    count = len(os.listdir(data_dir))
    for i, dataset in enumerate(os.listdir(data_dir)):
        data = np.loadtxt(data_dir+dataset, delimiter='\t', usecols=(0,1,2,3,4))

        mealnum = 0
        mealsize = 0
        for j in data:
            
            f_length, g_start, rate, p_length, g_end = j
            mealsize += 3.5*rate*f_length
            
            if p_length > cutoff and p_length < err_thresh:
                p_lengths.append(p_length/60) # convert to minutes
                mealsizes.append(mealsize)

                if mealnum == 0:
                	ratios.append(float(p_length/60)/mealsize) # convert to minutes
                	
                mealsize = 0
                c.append(float(i)/count)


                mealnum += 1

            if p_length > err_thresh:
                print 'Error in %s, IMI of %3.0f' %(dataset, p_length)

    ## Process to use in moving window
    results = zip(mealsizes, p_lengths)
    results = np.array(results)

    ## Generate moving window mean
    x_grid = np.linspace(0, max(mealsizes), 50)

    means = []
    for xval in x_grid:
        usevals = np.abs(results[:,0] - xval) < windowsize
        meanval = np.mean(results[usevals, 1])
        means.append(meanval)
        
    means = np.array(means)

    ## Plot moving window
    fig, axes = plt.subplots(1, figsize=(5,5))
    axes.scatter(results[:,0], 
                 results[:,1], 
                 alpha=0.3, 
                 c='b', 
                 cmap=cm.gist_ncar)
    
    axes.plot(x_grid, means, c='k')
    
    ## Calculate satiety ratio
    """
    srs = []
    for i in results:
        ratio = float(i[1])/i[0]
        print ratio
        srs.append(ratio)
    satiety_ratio = np.mean(srs)
    """
    #satiety_ratio = np.mean(results, axis=0)[1]/np.mean(results, axis=0)[0]
    #satiety_ratio = float(results[0][1])/results[0][0]
    satiety_ratio = np.mean(ratios)
    print satiety_ratio

    
    ratio_predict = x_grid*satiety_ratio
    axes.plot(x_grid, ratio_predict, c='r')
    
    return fig, axes

def IMI_inset(theta7, theta8, num_samples=100, figsize=(2,2), c='b'):
	fig, axes = plt.subplots(1, figsize=figsize)
	k1 = 0.00055

	x_vals = np.linspace(0, 30, 10)
	y = []
	for x in x_vals:
		samples = []
		for i in range(num_samples):
			samples.append(fl.sample_L(x, k1, theta7, theta8)/60) # convert to minutes

		y.append(np.mean(samples))

	axes.plot(x_vals, y, c=c)
	return fig, axes

"""
Figure 4
"""
def termination_prob(data_dict):
	fig, axes = plt.subplots(1, figsize=(5,5))

	x_vals = np.linspace(0, 20,100)
	for key in data_dict.keys():
		post = data_dict[key]
		theta4 = post[:,3]
		theta5 = post[:,4]
		data = key.split('_')[:-1]
		c = helpers.get_colour(data)

		sig_min, sig_mean, sig_max, sig_int = helpers.get_Q(x_vals, theta4, theta5)

		axes.plot(x_vals, sig_mean, c=c)

	return fig, axes

def termination_given_params(theta4, theta5, figsize=(2,2)):
	fig, axes = plt.subplots(1, figsize=figsize)

	x_vals = np.linspace(0, 30, 1000)
	y = []
	for x in x_vals:
		y.append(fl.Q(x, theta4, theta5))

	axes.plot(x_vals, y, c='b')
	#axes.fill_between(x_vals, sig_min, sig_max, alpha=0.3)
	return fig, axes


def param_change_effect(data_dict, indivs, param_idx, delta, num_samples=100, duration=8*60*60):
	fig, axes = plt.subplots(len(indivs), 1, figsize=(5,5))

	for i, indiv in enumerate(indivs):
		post = data_dict[indiv]
		post = np.mean(post, axis=0)
		perturbed_params = np.copy(post)
		perturbed_params[param_idx] = perturbed_params[param_idx] + delta
		
		baseline_samples = []
		perturbed_samples = []
		for j in range(num_samples):
			baseline_samples.append(3600*fs.sample(duration, post, 0)[1]/duration)
			perturbed_samples.append(3600*fs.sample(duration, perturbed_params, 0)[1]/duration)

		axes[i].hist(baseline_samples, color='b', bins=20, normed=True, alpha=0.6)
		axes[i].hist(perturbed_samples, color='r', bins=20, normed=True, alpha=0.6)

	return fig, axes

def pairwise_param_changes(data_dict, indiv, delta=0.1, num_samples=100, duration=8*60*60):
	fig, axes = plt.subplots(1, figsize=(5,5))

	post = data_dict[indiv]
	post = np.mean(post, axis=0)

	num_vars = len(post)
	delta_mat = np.zeros((num_vars, num_vars))

	for i in range(num_vars):
		for j in range(num_vars):
			perturbed_params = np.copy(post)
			perturbed_params[i] = perturbed_params[i] + delta
			perturbed_params[j] = perturbed_params[j] + delta

			samples = []
			for k in range(num_samples):
				samples.append(fs.sample(duration, perturbed_params, 0)[1])

			delta_mat[i,j] = np.mean(samples)

	axes.matshow(delta_mat)

	return fig, axes


def meals_from_data(data, cutoff=300):
    mealnum = 0
    mealsize = 0
    mealdur = 0

    mealsizes = []
    mealdurs = []
    p_lengths = []
    for event in data:
        
        f_length, g_start, rate, p_length, g_end = event
        mealsize += 3.5*rate*f_length
        mealdur += f_length
        
        if p_length > cutoff:
            p_lengths.append(p_length)

            mealsizes.append(mealsize)
            mealsize = 0

            mealdurs.append(mealdur)
            mealdur = 0

            mealnum += 1

    return mealsizes, mealdurs, p_lengths

def meal_stats_from_delta(data_dict, indiv, param_idxs, deltas, num_samples=100, duration=8*60*60, figsize=(10, 5)):
	fig, axes = plt.subplots(1, 5, figsize=figsize)

	post = data_dict[indiv]
	post = np.mean(post, axis=0)
	data = indiv.split('_')
	c = helpers.get_colour(data[:4])

	## Stats from perturbed posterior
	perturbed_params = np.copy(post)
	for i, param_idx in enumerate(param_idxs):
		perturbed_params[param_idx] = perturbed_params[param_idx] + deltas[i]

	baseline_amts = []
	baseline_sizes = []
	baseline_durs = []
	baseline_IMIs = []
	baseline_counts = []

	perturbed_amts = []
	perturbed_sizes = []
	perturbed_durs = []
	perturbed_IMIs = []
	perturbed_counts = []

	## TODO: finish

	for i in range(num_samples):
		## Sample from baseline
		sample_data = fs.sample(duration, post, 0)
		sample_amount = sample_data[1]
		events = sample_data[-1]
		baseline_amts.append(3600*sample_amount/duration)

		mealsizes, mealdurs, p_lengths = meals_from_data(events)

		baseline_sizes += mealsizes
		baseline_durs += mealdurs
		baseline_IMIs += p_lengths
		baseline_counts.append(len(p_lengths))

		## Sample from perturbed
		sample_data = fs.sample(duration, perturbed_params, 0)
		sample_amount = sample_data[1]
		events = sample_data[-1]
		perturbed_amts.append(3600*sample_amount/duration)

		mealsizes, mealdurs, p_lengths = meals_from_data(events)

		perturbed_sizes += mealsizes
		perturbed_durs += mealdurs
		perturbed_IMIs += p_lengths
		perturbed_counts.append(len(p_lengths))

	## Convert to minutes
	baseline_IMIs = np.array(baseline_IMIs)
	baseline_IMIs = baseline_IMIs/60.
	baseline_durs = np.array(baseline_durs)
	baseline_durs = baseline_durs/60.

	perturbed_IMIs = np.array(perturbed_IMIs)
	perturbed_IMIs = perturbed_IMIs/60.
	perturbed_durs = np.array(perturbed_durs)
	perturbed_durs = perturbed_durs/60.

	## Bar plots for baseline
	axes[0].bar(0, np.mean(baseline_amts), color='b', label='$\\theta_{0}$')
	axes[1].bar(0, np.mean(baseline_sizes), color='b')
	axes[2].bar(0, np.mean(baseline_durs), color='b')
	axes[3].bar(0, np.mean(baseline_IMIs), color='b')
	axes[4].bar(0, np.mean(baseline_counts), color='b')

	## Bar plots for perturbed
	axes[0].bar(1, np.mean(perturbed_amts), color='r', label='$\Delta \\theta$')
	axes[1].bar(1, np.mean(perturbed_sizes), color='r')
	axes[2].bar(1, np.mean(perturbed_durs), color='r')
	axes[3].bar(1, np.mean(perturbed_IMIs), color='r')
	axes[4].bar(1, np.mean(perturbed_counts), color='r')

	axes[0].set_xticklabels([])
	axes[1].set_xticklabels([])
	axes[2].set_xticklabels([])
	axes[3].set_xticklabels([])
	axes[4].set_xticklabels([])

	axes[0].legend()

	return fig, axes







def param_delta_curve(data_dict, indivs, param_idx, delta_range, num_samples=100, duration=8*60*60, figsize=(5,10)):
	fig, axes = plt.subplots(5, 1, figsize=figsize)

	for indiv in indivs:
		post = data_dict[indiv]
		post = np.mean(post, axis=0)
		data = indiv.split('_')
		c = helpers.get_colour(data[:4])

		y1 = []
		y1_min = []
		y1_max = []

		y2 = []
		y2_min = []
		y2_max = []

		y3 = []
		y3_min = []
		y3_max = []

		y4 = []
		y4_min = []
		y4_max = []

		y5 = []
		y5_min = []
		y5_max = []

		for delta in delta_range:
			perturbed_params = np.copy(post)
			perturbed_params[param_idx] = perturbed_params[param_idx] + delta

			sample_amts = []
			sample_sizes = []
			sample_durs = []
			sample_IMIs = []
			meal_counts = []
			for i in range(num_samples):
				sample_data = fs.sample(duration, perturbed_params, 0)
				sample_amount = sample_data[1]
				events = sample_data[-1]
				sample_amts.append(3600*sample_amount/duration)

				mealsizes, mealdurs, p_lengths = meals_from_data(events)

				sample_sizes += mealsizes
				sample_durs += mealdurs
				sample_IMIs += p_lengths
				meal_counts.append(len(p_lengths))

			## Normalised feeding amount distribution
			y1.append(np.mean(sample_amts))
			#y1_min.append(np.percentile(sample_amts, 5))
			#y1_max.append(np.percentile(sample_amts, 95))

			## Meal size
			y2.append(np.mean(sample_sizes))
			#y2_min.append(np.percentile(sample_sizes, 5))
			#y2_max.append(np.percentile(sample_sizes, 95))

			## Meal duration
			y3.append(np.mean(sample_durs))
			#y3_min.append(np.percentile(sample_durs, 5))
			#y3_max.append(np.percentile(sample_durs, 95))

			## Intermeal interval
			y4.append(np.mean(sample_IMIs))
			#y4_min.append(np.percentile(sample_IMIs, 5))
			#y4_max.append(np.percentile(sample_IMIs, 95))			

			## Meal count
			#print meal_counts	
			y5.append(np.mean(meal_counts))
			#y5_min.append(np.percentile(meal_counts, 5))
			#y5_max.append(np.percentile(meal_counts, 95))	

		axes[0].plot(delta_range, y1, c=c, marker='o')
		#axes[0].fill_between(delta_range, y1_min, y1_max, alpha=0.1, color=c)

		axes[1].plot(delta_range, y2, c=c, marker='o')
		#axes[1].fill_between(delta_range, y2_min, y2_max, alpha=0.1, color=c)

		axes[2].plot(delta_range, y3, c=c, marker='o')
		#axes[2].fill_between(delta_range, y3_min, y3_max, alpha=0.1, color=c)

		axes[3].plot(delta_range, y4, c=c, marker='o')
		#axes[3].fill_between(delta_range, y4_min, y4_max, alpha=0.1, color=c)

		axes[4].plot(delta_range, y5, c=c, marker='o')
		#axes[4].fill_between(delta_range, y5_min, y5_max, alpha=0.1, color=c)

	return fig, axes



"""
Figure 5
"""
def dosing_protocol(data_dict, protocol, num_samples=10, cutoff=300):
	fig, axes = plt.subplots(1, figsize=(5,5))

	## Do the sampling
	amounts, ts_mean, ts_pc, last_ts = helpers.sample_protocol(data_dict, protocol, num_samples, cutoff)

	## Plot mean time series and bounds
	xax = np.arange(len(ts_mean))
	axes.plot(ts_mean, c='r')
	axes.fill_between(xax,
					  ts_pc[0], 
					  ts_pc[1],
					  color='r',
					  alpha=0.3)

	## Plot a sample time series (the last one)
	axes.plot(last_ts)

	## Plot the protocol bars
	## TODO: make for general protocols
	protocol1 = np.arange(protocol[0][0]*3600)
	bar = np.zeros(protocol1.shape[0])
	axes.plot(protocol1, bar, c='k')

	print np.mean(amounts), np.std(amounts)

	return fig, axes

def optimise_protocols(data_dict, druglist, protocol_size, duration, min_default=2, num_samples=10, cutoff=300, pc=5):
	## NOTE: default drug is the first one in the list
	fig, axes = plt.subplots(3, 1, figsize=(5,10))

	## Create protocol list
	protocol_list = helpers.make_protocol_list(druglist, protocol_size, duration, min_default)

	## Iterate over protocols
	mean_amounts = []
	all_amounts = []
	time_series = []
	for protocol in protocol_list:
		amounts, ts_mean, ts_pc, last_ts = helpers.sample_protocol(data_dict, protocol, num_samples, cutoff, 5)
		mean_amounts.append(np.mean(amounts))
		time_series.append((ts_mean, ts_pc))
		all_amounts.append(amounts)

	## Plot policy ranking
	protocols_to_rank = zip(mean_amounts, protocol_list)
	ranked_protocols = sorted(protocols_to_rank, key=lambda x:x[0])
	protocol_info = ranked_protocols
	xax = np.arange(len(mean_amounts))
	amounts_to_plot = [i[0] for i in ranked_protocols] # extract amount eaten
	amounts_to_plot = np.array(amounts_to_plot)/(1.*duration*protocol_size) # convert to kcal/hr
	axes[0].scatter(xax, amounts_to_plot) # could do errorbar for SEM if necessary/useful

	## Plot optimal & pessimal protocol stomach fullness
	protocols_to_rank = zip(mean_amounts, time_series)
	ranked_protocols = sorted(protocols_to_rank, key=lambda x:x[0])
	opt_protocol = ranked_protocols[0]
	pess_protocol = ranked_protocols[-1]

	## This is a bit hacky but works OK
	opt_mean = opt_protocol[1][0]
	opt_pc = opt_protocol[1][1]
	opt_pc_low = opt_pc[0]
	opt_pc_high = opt_pc[1]

	pess_mean = pess_protocol[1][0]
	pess_pc = pess_protocol[1][1]
	pess_pc_low = pess_pc[0]
	pess_pc_high = pess_pc[1]
	
	xax = np.arange(len(opt_mean))
	xax = xax/60.

	axes[1].plot(xax, opt_mean, c='b')
	axes[1].fill_between(xax,
				  		 opt_pc_low, 
				  		 opt_pc_high,
				  		 color='b',
				  		 alpha=0.3)

	axes[1].plot(xax, pess_mean, c='r')
	axes[1].fill_between(xax,
			  		 	 pess_pc_low, 
			  			 pess_pc_high,
			  			 color='r',
			  			 alpha=0.3)

	## Plot amount distributions for optimal & pessimal protocols
	protocols_to_rank = zip(mean_amounts, all_amounts)
	ranked_protocols = sorted(protocols_to_rank, key=lambda x:x[0])
	opt_protocol = ranked_protocols[0]
	pess_protocol = ranked_protocols[-1]
	axes[2].hist(opt_protocol[1], color='b', bins=20, normed=True, alpha=0.6)
	axes[2].hist(pess_protocol[1], color='r', bins=20, normed=True, alpha=0.6)

	return fig, axes, protocol_info

def behav_change_effect_group(data_dict, groupname, xmax, num_samples=100, duration=8*60*60):
	fig, axes = plt.subplots(2, 1, figsize=(5,5))

	post = data_dict[groupname]
	post = np.mean(post, axis=0)
	
	baseline_samples = []
	perturbed_samples = []
	for i in range(num_samples):
		baseline_samples.append(fs.sample(duration, post, 0)[1])
		perturbed_samples.append(fs.sample_lim_x(duration, post, 0, xmax)[1])

	baseline_ts = fs.sample(duration, post, 0)[0]
	perturbed_ts = fs.sample_lim_x(duration, post, 0, xmax)[0]

	axes[0].plot(baseline_ts[:duration], c='b')
	axes[0].plot(perturbed_ts[:duration], c='r')
	axes[0].axhline(xmax, c='k', ls='--')

	axes[1].hist(baseline_samples, color='b', bins=20, normed=True)
	axes[1].hist(perturbed_samples, color='r', bins=20, normed=True)

	return fig, axes

def behav_change_effect_indiv(df, xmax, thetas, num_samples=100, duration=8*60*60):
	fig, axes = plt.subplots(8,1, figsize=(5,5))

	def calc_change(row):
		post = row[thetas]

		true_val = row['rate']
		c = row['drug_c']
		ms = row['ms']
		x0 = float(row['x0'])

		samples = []
		for i in range(num_samples):
			samples.append(fs.sample_lim_x(duration, post, x0, xmax)[1])

		delta = 3600.*np.mean(samples)/duration - true_val

		for i in range(0, 8):
			axes[i].scatter(post[i], delta, c=c, marker=ms)

	## Iterate the plotter over the dataframe
	df.apply(calc_change, axis=1)

	return fig, axes

def behav_response_curve(data_dict, indivs, delta_range, num_samples=100, duration=8*60*60, figsize=(5,10)):
	fig, axes = plt.subplots(5, 1, figsize=figsize)

	for indiv in indivs:
		post = data_dict[indiv]
		post = np.mean(post, axis=0)
		data = indiv.split('_')
		c = helpers.get_colour(data[:4])

		y1 = []
		y1_min = []
		y1_max = []

		y2 = []
		y2_min = []
		y2_max = []

		y3 = []
		y3_min = []
		y3_max = []

		y4 = []
		y4_min = []
		y4_max = []

		y5 = []
		y5_min = []
		y5_max = []

		for delta in delta_range:
			sample_amts = []
			sample_sizes = []
			sample_durs = []
			sample_IMIs = []
			meal_counts = []
			for i in range(num_samples):
				sample_data = fs.sample_lim_x(duration, post, 0, delta)
				sample_amount = sample_data[1]
				events = sample_data[-1]
				sample_amts.append(3600*sample_amount/duration)

				mealsizes, mealdurs, p_lengths = meals_from_data(events)

				sample_sizes += mealsizes
				sample_durs += mealdurs
				sample_IMIs += p_lengths
				meal_counts.append(len(p_lengths))

			## Normalised feeding amount distribution
			y1.append(np.mean(sample_amts))
			#y1_min.append(np.percentile(sample_amts, 5))
			#y1_max.append(np.percentile(sample_amts, 95))

			## Meal size
			y2.append(np.mean(sample_sizes))
			#y2_min.append(np.percentile(sample_sizes, 5))
			#y2_max.append(np.percentile(sample_sizes, 95))

			## Meal duration
			y3.append(np.mean(sample_durs))
			#y3_min.append(np.percentile(sample_durs, 5))
			#y3_max.append(np.percentile(sample_durs, 95))

			## Intermeal interval
			y4.append(np.mean(sample_IMIs))
			#y4_min.append(np.percentile(sample_IMIs, 5))
			#y4_max.append(np.percentile(sample_IMIs, 95))			

			## Meal count
			#print meal_counts	
			y5.append(np.mean(meal_counts))
			#y5_min.append(np.percentile(meal_counts, 5))
			#y5_max.append(np.percentile(meal_counts, 95))	

		axes[0].plot(delta_range, y1, c=c, marker='o')
		#axes[0].fill_between(delta_range, y1_min, y1_max, alpha=0.1, color=c)

		axes[1].plot(delta_range, y2, c=c, marker='o')
		#axes[1].fill_between(delta_range, y2_min, y2_max, alpha=0.1, color=c)

		axes[2].plot(delta_range, y3, c=c, marker='o')
		#axes[2].fill_between(delta_range, y3_min, y3_max, alpha=0.1, color=c)

		axes[3].plot(delta_range, y4, c=c, marker='o')
		#axes[3].fill_between(delta_range, y4_min, y4_max, alpha=0.1, color=c)

		axes[4].plot(delta_range, y5, c=c, marker='o')
		#axes[4].fill_between(delta_range, y5_min, y5_max, alpha=0.1, color=c)

	return fig, axes

def refractory_period(data_dict, indivs, period_range, num_samples=100, duration=8*60*60, figsize=(5,10)):
	fig, axes = plt.subplots(3, 1, figsize=figsize)
	fig2, axes2 = plt.subplots(2, 1, figsize=(5,5))

	for indiv in indivs:
		print 'Generating samples for %s' %(indiv)
		post = data_dict[indiv]
		post = np.mean(post, axis=0)
		data = indiv.split('_')
		c = helpers.get_colour(data[:4])

		y1 = []
		y1_min = []
		y1_max = []

		y2 = []
		y2_min = []
		y2_max = []

		y3 = []
		y3_min = []
		y3_max = []

		y4 = []
		y4_min = []
		y4_max = []

		y5 = []
		y5_min = []
		y5_max = []

		for period in period_range:
			sample_amts = []
			sample_sizes = []
			sample_durs = []
			sample_IMIs = []
			meal_counts = []
			for i in range(num_samples):
				sample_data = fs.sample_refractory(duration, post, 0, period)
				sample_amount = sample_data[1]
				events = sample_data[-1]
				sample_amts.append(3600*sample_amount/duration)

				mealsizes, mealdurs, p_lengths = meals_from_data(events)

				sample_sizes += mealsizes
				sample_durs += mealdurs
				sample_IMIs += p_lengths
				meal_counts.append(len(p_lengths))

			## Normalised feeding amount distribution
			y1.append(np.mean(sample_amts))
			#y1_min.append(np.percentile(sample_amts, 5))
			#y1_max.append(np.percentile(sample_amts, 95))

			## Meal size
			y2.append(np.mean(sample_sizes))
			#y2_min.append(np.percentile(sample_sizes, 5))
			#y2_max.append(np.percentile(sample_sizes, 95))

			## Meal duration
			y3.append(np.mean(sample_durs)/60.)
			#y3_min.append(np.percentile(sample_durs, 5))
			#y3_max.append(np.percentile(sample_durs, 95))

			## Intermeal interval
			y4.append(np.mean(sample_IMIs)/60.) # convert to minutes
			#y4_min.append(np.percentile(sample_IMIs, 5))
			#y4_max.append(np.percentile(sample_IMIs, 95))			

			## Meal count
			#print meal_counts	
			y5.append(np.mean(meal_counts))
			#y5_min.append(np.percentile(meal_counts, 5))
			#y5_max.append(np.percentile(meal_counts, 95))

			if period == period_range[0]:
				axes2[0].hist(np.array(sample_IMIs)/60, color=c, alpha=0.6, bins=20, normed=True)

			if period == period_range[-1]:
				axes2[1].hist(np.array(sample_IMIs)/60, color=c, alpha=0.6, bins=20, normed=True)
		"""
		axes[0].plot(period_range/60., y5, c=c, marker='o')
		#axes[0].fill_between(delta_range, y1_min, y1_max, alpha=0.1, color=c)

		axes[1].plot(period_range/60., y2, c=c, marker='o')
		#axes[1].fill_between(delta_range, y2_min, y2_max, alpha=0.1, color=c)
		"""
		axes[0].plot(period_range/60., y3, c=c, marker='o')
		#axes[2].fill_between(delta_range, y3_min, y3_max, alpha=0.1, color=c)

		axes[1].plot(period_range/60., y1, c=c, marker='o')
		#axes[3].fill_between(delta_range, y4_min, y4_max, alpha=0.1, color=c)

		axes[2].plot(period_range/60., y4, c=c, marker='o')
		#axes[4].fill_between(delta_range, y5_min, y5_max, alpha=0.1, color=c)

	return fig, axes, fig2, axes2

def power_study_false_pos(trace, num_repeats, duration, figsize=(5,5)):
	fig, axes = plt.subplots(1, figsize=figsize)
	
	chol = trace['chol_cov']
	means = trace['mu']

	trace_size = means.shape[0]
	samplesizes = np.arange(5, 50, 5)

	p_thresh = 0.05

	false_pos = []
	for samplesize in samplesizes:
		num_fp = 0
		for i in range(num_repeats):
			## Generate group-level parameters by sampling posterior
			use_idx = np.random.randint(trace_size)
			use_chol = chol[use_idx, :]
			cov = helpers.cov_from_chol(8, use_chol)
			mu = means[use_idx, :]

			group1 = helpers.sample_group(mu, cov, samplesize, duration)
			group2 = helpers.sample_group(mu, cov, samplesize, duration)

			tstat, pval = scipy.stats.ttest_ind(group1, group2, equal_var=False)

			if pval/2 < p_thresh and tstat > 0:
				num_fp += 1.

		false_pos.append(num_fp/num_repeats)

	axes.plot(samplesizes, false_pos)


	return fig, axes

def power_study_false_neg(trace1, trace2, num_repeats, duration, figsize=(5,5)):
	fig, axes = plt.subplots(1, figsize=figsize)
	
	chol1 = trace1['chol_cov']
	means1 = trace1['mu']

	chol2 = trace2['chol_cov']
	means2 = trace2['mu']

	trace_size = means1.shape[0]
	samplesizes = np.arange(5, 56, 10)

	p_thresh = 0.05

	false_neg = []
	for samplesize in samplesizes:
		num_accept = 0
		for i in range(num_repeats):
			## Generate group-level parameters by sampling posterior
			use_idx = np.random.randint(trace_size)
			use_chol1 = chol1[use_idx, :]
			cov1 = helpers.cov_from_chol(8, use_chol1)
			mu1 = means1[use_idx, :]

			use_chol2 = chol2[use_idx, :]
			cov2 = helpers.cov_from_chol(8, use_chol2)
			mu2 = means2[use_idx, :]

			group1 = helpers.sample_group(mu1, cov1, samplesize, duration)
			group2 = helpers.sample_group(mu2, cov2, samplesize, duration)

			tstat, pval = scipy.stats.ttest_ind(group1, group2, equal_var=False)

			if pval/2 < p_thresh and tstat > 0:
				num_accept += 1.

		false_neg.append(100*(num_repeats - num_accept)/num_repeats)

	axes.plot(samplesizes, false_neg, label='Model-based')

	return fig, axes

def power_study_multi_comp(traces, num_repeats, duration, figsize=(5,5)):
	samplesizes = np.arange(5,56,10)
	p_thresh = 0.05

	false_neg = []
	for samplesize in samplesizes:
		num_accept = 0
		for i in range(num_repeats):
			## Generate samples from each trace
			samples = []
			for trace in traces:
				chol = trace['chol_cov']
				means = trace['mu']
				trace_size = means.shape[0]

				use_idx = np.random.randint(trace_size)
				use_chol = chol[use_idx, :]
				cov = helpers.cov_from_chol(8, use_chol)
				mu = means[use_idx, :]

				group_sample = helpers.sample_group(mu, cov, samplesize, duration)

				samples.append(group_sample)

			## ANOVA between groups
			f, p = scipy.stats.f_oneway(samples[1], samples[0])
			if p < p_thresh:
				num_accept += 1

		false_neg.append(100.*(num_repeats - num_accept)/num_repeats)

	fig, axes = plt.subplots(1)
	axes.plot(samplesizes, false_neg)

	return fig, axes


			