import numpy as np
from scipy import integrate
import sys
from datetime import datetime, timedelta
import csv
import CLAMS_parsers as parser
import ODEs as odes

def get_ts(bout_data):
    k1 = 0.00055
    """
    Read in the data and digestion rate
    """
    exp_start = bout_data[0][0]

    """
    Now do the integrating
    """
    events = []
    gut_ts_holder = []
    g_start = 0.0 # initialise to 0.0     
    for i, bout in enumerate(bout_data):
        #print bout
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

            if (t_start.hour < d_start or t_start.hour >= d_end):
                period = 'D' # dark period
            else:
                period = 'L' # light period

            
            feeding_interval = np.linspace(0, f_length, f_length) # want at 1s resolution 
            pause_interval = np.linspace(0, p_length, p_length) # as above
            
            rate = float(bout[2])/f_length
            
            """
            Now solve ODE on feeding
            """
            g_feeding = integrate.odeint(odes.feeding_ode, g_start, feeding_interval, args=(rate,))
            g_feeding = g_feeding.clip(min=0) # remove the small numerical errors bring it < 0
            g_end_feeding = g_feeding[:,0][-1]

            gut_ts_holder.append(np.stack(g_feeding, axis=1)[0]) # need to get the ts as an array
            
            g_digestion = integrate.odeint(odes.digestion_ode, g_end_feeding, pause_interval, args=(k1,))
            g_digestion = g_digestion.clip(min=0) # remove the small numerical errors bring it < 0
            g_end = g_digestion[:,0][-1]

            gut_ts_holder.append(np.stack(g_digestion, axis=1)[0])

            """
            Store time from start for conditional export
            """
            t_from_start = float((t_start - exp_start).total_seconds())/3600.
            
            """
            Append
            """
            event = [f_length, g_start, rate, p_length, g_end_feeding, period, t_from_start]
            events.append(event)
            
            """
            Setup for the next go round the loop
            """
            g_start = g_end


    gut_ts = np.concatenate(gut_ts_holder)

    return gut_ts

def get_events(bout_data):
    k1 = 0.00055
    """
    Read in the data and digestion rate
    """
    exp_start = bout_data[0][0]

    """
    Now do the integrating
    """
    events = []
    gut_ts_holder = []
    g_start = 0.0 # initialise to 0.0     
    for i, bout in enumerate(bout_data):
        #print bout
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

            if (t_start.hour < d_start or t_start.hour >= d_end):
                period = 'D' # dark period
            else:
                period = 'L' # light period

            #print t_start, t_end, t_next, f_length, p_length
            
            feeding_interval = np.linspace(0, f_length, f_length) # want at 1s resolution 
            pause_interval = np.linspace(0, p_length, p_length) # as above
            
            rate = float(bout[2])/f_length
            
            """
            Now solve ODE on feeding
            """
            ## Feeding
            g_end_feeding = g_start + 3.5*rate*f_length # calorie content 3.5 kcal/g
            
            ## Pausing 
            t_c = 2.*np.sqrt(g_end_feeding)/k1

            if p_length <= t_c:
                g_end = 0.25*np.power((2.*np.sqrt(g_end_feeding) - k1*p_length), 2)

            else:
                g_end = 0

            """
            Store time from start for conditional export
            """
            t_from_start = float((t_start - exp_start).total_seconds())/3600.
            
            """
            Append
            """
            event = [f_length, 
                     g_start, 
                     rate, 
                     p_length, 
                     g_end_feeding, 
                     period, 
                     t_from_start, 
                     bout[0], 
                     bout[1], 
                     bout[2]]

            events.append(event)
            
            """
            Setup for the next go round the loop
            """
            g_start = g_end

    return events