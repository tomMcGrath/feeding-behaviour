import numpy as np
from scipy import integrate
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import csv
import pandas as pd

def parse_bouts(file_path):
	"""
	Returns a pandas timeseries object containing food intake data calculated
	from the feeding bouts, as well as subject data (numeric and subject ID,
	mass in g).
	:param file_path: the file containing feeding bout data
	"""
	data_file = open(file_path, 'r')
	data_reader = csv.reader(data_file)

	in_data = False
	headers = []
	preproc_data = []
	for line in data_reader:
		if in_data == False:
			"""
			The faffing below is necessary because CLAMS for some reason
			produces inconsistent header files in different runs.
			"""
			if len(line) == 0:
				continue

			if line[0] == ':DATA':
				in_data = True
				continue

			else:
				if len(line) == 1:
					line = line[0].split(':')

				headers.append(line)

		elif in_data == True:
			preproc_data.append(line)

		else:
			print 'Error in CLAMS parsing'

	"""
	Pull out header data
	"""

	num_id = headers[3][1]
	subject_id = headers[4][1]
	mass = headers[5][1]


	data = []
	for i in preproc_data[4:-1]:
		start = datetime.strptime((i[3] + ' ' + i[4]).strip(' '), '%d/%m/%Y %H:%M:%S')
		end = datetime.strptime((i[3] + ' ' + i[5]).strip(' '), '%d/%m/%Y %H:%M:%S')
		amount_eaten = float(i[7])

		if amount_eaten <= 0.0: # this is a spurious bout
			continue

		if start.hour > end.hour: # the bout has spanned midnight
			end = end + timedelta(days=1)

		data.append((start, end, amount_eaten))

	return num_id, subject_id, mass, data

"""
Now integrate the ODEs
"""
def feeding_ode(x, t, k2):
	"""
	ODE for feeding with rate k2
	"""
	calorie_content = 3.5 # kcal/gram

	return k2*calorie_content

def digestion_ode(x, t):
	"""
	ODE for feeding with rate k1
	"""
	k1 = 0.00055 # digestion rate constant
	if x > 0:
		ans = -k1*np.sqrt(x)
	else:
		ans = 0.0
	return ans

def plot_ts(num_id, subject_id, mass, bout_data):
	"""
	Now do the integrating
	"""
	events = []
	g_start = 0.0 # initialise to 0.0
	g_ts = np.array([])
	f_ts = np.array([])

	starttime = bout_data[0][0]
	endtime = bout_data[-1][1]
	starttimes = []
	cutoff = 1800 # arbitrary meal cutoff

	for i, bout in enumerate(bout_data):
		if i != len(bout_data) - 1:
			"""
			Record feeding -> pause -> feeding cycle (this is what we need
			to assign to short or long pause)
			Format:
			(feeding length, stomach at start of feeding, feeding rate,
			pause length, stomach at start of pause)
			"""
			t_start = bout[0]
			t_end = bout[1]
			f_length = (t_end - t_start).total_seconds()
			t_next = bout_data[i+1][0]
			p_length = (t_next - t_end).total_seconds()
			d_start = 6
			d_end = 18

			if (t_start.hour < d_start or t_start.hour > d_end):
				period = 'D' # dark period
			else:
				period = 'L' # light period


			feeding_interval = np.linspace(0, f_length, f_length) # want at 1s resolution
			pause_interval = np.linspace(0, p_length, p_length) # as above

			rate = float(bout[2])/f_length

			"""
			Now solve ODE on feeding
			"""
			g_feeding = integrate.odeint(feeding_ode, g_start, feeding_interval, args=(rate,))
			g_feeding = g_feeding.clip(min=0) # remove the small numerical errors bring it < 0
			g_end_feeding = g_feeding[:,0][-1]

			g_digestion = integrate.odeint(digestion_ode, g_end_feeding, pause_interval)
			g_digestion = g_digestion.clip(min=0) # remove the small numerical errors bring it < 0
			g_end = g_digestion[:,0][-1]


			"""
			Append
			"""
			## Events
			event = [f_length, g_start, rate, p_length, g_end_feeding, period]
			events.append(event)

			## Gut time series
			g_ts = np.hstack([g_ts, g_feeding[:,0]])
			g_ts = np.hstack([g_ts, g_digestion[:,0]])

			## Bout time series
			f_ts = np.hstack([f_ts, rate*np.ones(len(g_feeding[:,0]))])
			f_ts = np.hstack([f_ts, np.zeros(len(g_digestion[:,0]))])

			## Start times for start-of-meal
			if events[i-1][3] > cutoff:
				starttimes.append(t_start)

			"""
			Setup for the next go round the loop
			"""
			g_start = g_end

	"""
	Plot the graph of gut fullness
	"""
	print len(g_ts)
	print (endtime-starttime).total_seconds()
	print len(bout_data)

	times = pd.date_range(starttime, endtime, freq='s')

	g_ts = pd.Series(g_ts, index = times[0:len(g_ts)])
	f_ts = pd.Series(f_ts, index = times[0:len(f_ts)])

	## Setup plot
	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)

	f_ts.plot(ax=ax, c='k')
	g_ts.plot(ax=ax2, c='k')

	## Dark period shading
	lightstart = starttime.replace(hour=6, minute=0, second=0)
	darkstart = starttime.replace(hour=18, minute=0, second=0)

	alphaval=0.2
	ax.axvspan(darkstart, lightstart+timedelta(days=1), color='b', alpha=alphaval)
	ax.axvspan(darkstart+timedelta(days=1), lightstart+timedelta(days=2), color='b', alpha=alphaval)
	ax.axvspan(darkstart+timedelta(days=2), lightstart+timedelta(days=3), color='b', alpha=alphaval)
	ax.axvspan(darkstart+timedelta(days=3), lightstart+timedelta(days=4), color='b', alpha=alphaval)

	ax2.axvspan(darkstart, lightstart+timedelta(days=1), color='b', alpha=alphaval)
	ax2.axvspan(darkstart+timedelta(days=1), lightstart+timedelta(days=2), color='b', alpha=alphaval)
	ax2.axvspan(darkstart+timedelta(days=2), lightstart+timedelta(days=3), color='b', alpha=alphaval)
	ax2.axvspan(darkstart+timedelta(days=3), lightstart+timedelta(days=4), color='b', alpha=alphaval)

	## Bout start indicators
	"""
	for time in starttimes:
		print time
		ax2.axvline(time, c='r', ls='--')
	"""
	## Axis labels
	ax.set_ylabel('Food intake (kcal/s)')
	ax2.set_ylabel('Stomach fullness (kcal)')

	plt.show()

def ts_from_bouts(bout_data):
	k1 = 0.00055

	gut_ts_holder = []
	bout_ts_holder = []

	for bout in bout_data:
		f_length, g_start, rate, p_length, g_end_feeding = bout

		f_times = np.arange(f_length)
		p_times = np.arange(p_length)

		## Feeding ts
		f_ts = g_start + 3.5*rate*f_times # 3.5 kcal/g
		feed_bout = rate*np.ones(int(f_length))
		gut_ts_holder.append(f_ts)
		bout_ts_holder.append(feed_bout)

		## Pausing ts
		t_c = 2.*np.sqrt(g_end_feeding)/k1

		if p_length <= t_c:
			pause_bout = np.zeros(int(p_length))
			p_ts = 0.25*np.power((2.*np.sqrt(g_end_feeding) - k1*p_times), 2)
			p_ts = np.clip(p_ts, 0, None)

		else:
			pause_bout = np.zeros(int(p_length))
			pause_idx = int(np.floor(t_c))
			p_ts_pause = 0.25*np.power((2.*np.sqrt(g_end_feeding) - k1*p_times[:pause_idx]), 2)

			p_ts_empty = np.zeros(int(p_length) - pause_idx)
			p_ts = np.concatenate((p_ts_pause, p_ts_empty))

		gut_ts_holder.append(p_ts)
		bout_ts_holder.append(pause_bout)

	gut_ts = np.concatenate(gut_ts_holder)
	bout_ts = np.concatenate(bout_ts_holder)

	fig, axes = plt.subplots(2,1)

	axes[0].plot(bout_ts)
	axes[1].plot(gut_ts)
	plt.show()
