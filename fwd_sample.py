import numpy as np
import fwd_likelihoods as ll
from scipy import integrate
import matplotlib.pyplot as plt

def sample(tmax, theta, x0, init_state='F'):
    theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8 = theta

    theta1 = np.power(10., theta1)
    theta2 = np.power(10., theta2)
    theta3 = np.power(10., theta3)
    theta6 = np.power(10., theta6)
    theta7 = np.power(10., theta7)
    theta8 = np.power(10., theta8)

    x = np.array([x0])
    amount_eaten = 0.0
    k2 = 0.00055 # Digestion rate parameter
    t = 0.0
    state = init_state

    ## Use if starting in pause state
    f_length = []
    g_start = x0
    rate = theta3

    num_events = 0
    bout_lengths = []
    short_lengths = []
    long_lengths = []
    events = []
    while t < tmax or state == 'L' or state == 'S':
        ## Feeding code
        if state == 'F':
            ## Get duration and rate
            rate = np.random.normal(theta2, theta3)
            while rate < 0: # ensure rate is positive
                rate = np.random.normal(theta2, theta3)

            if rate <= 0:
                print rate

            t_plus = np.random.exponential(1./theta1)
            t_plus = np.ceil(t_plus)

            if t_plus == 0:
                print 't=0'

            ## Integrate forward
            g_start = x[-1]

            predicted_x = g_start + t_plus*3.5*rate

            f_length = np.arange(t_plus) + 1
            x_segment = g_start + 3.5*rate*f_length

            if len(f_length) == 0:
                print t_plus

            if not np.isclose(predicted_x, x_segment[-1]):
                print predicted_x, x_segment[-1]

            ## Append results, increment t
            x = np.hstack((x, x_segment))
            t += t_plus
            amount_eaten += t_plus*rate
            num_events += 1
            bout_lengths.append(t_plus)

            ## Find next state through transition kernel
            Q_val = ll.Q(x[-1], theta4, theta5)
            u = np.random.uniform()

            if Q_val < u:
                state = 'S'

            else:
                state = 'L'

        ## Short pause code
        elif state == 'S':
            ## Get duration
            t_plus = np.random.exponential(1./theta6)
            t_plus = np.ceil(t_plus)

            ## Integrate forward
            g_end_feeding = x[-1]
            t_c = int(2.*np.sqrt(g_end_feeding)/k2)
            p_length = np.arange(t_plus)
            x_segment = 0.25*np.power(k2*p_length, 2) - np.sqrt(g_end_feeding)*k2*p_length + g_end_feeding
            x_segment = np.clip(x_segment, 0, None)

            if t_plus > t_c:
                x_segment[t_c:] = 0.

            ## Append results, increment t
            x = np.hstack((x, x_segment))
            t += t_plus
            short_lengths.append(t_plus)

            event = [len(f_length), g_start, rate, len(p_length), g_end_feeding]
            events.append(event)

            ## Switch back to feeding
            state = 'F'

        ## Long pause code
        elif state == 'L':
            ## Get duration
            t_plus = ll.sample_L(x[-1], k2, theta7, theta8)
            t_plus = int(t_plus)

            if t_plus > 12*60*60:
                t_plus = 12*60*60 ## avoid unreasonably long pauses leading to out of memory errors

            ## Integrate forward
            g_end_feeding = x[-1]
            t_c = int(2.*np.sqrt(g_end_feeding)/k2)

            p_length = np.arange(t_plus)
            x_segment = 0.25*np.power(k2*p_length, 2) - np.sqrt(g_end_feeding)*k2*p_length + g_end_feeding
            x_segment = np.clip(x_segment, 0, None)

            if t_plus > t_c:
                x_segment[t_c:] = 0.

            ## Append results, increment t
            x = np.hstack((x, x_segment))
            t += t_plus
            long_lengths.append(t_plus)

            event = [len(f_length), g_start, rate, len(p_length), g_end_feeding]
            events.append(event)

            ## Switch back to feeding
            state = 'F'

        else:
            print "ERROR, invalid state"

    return x, amount_eaten, num_events, bout_lengths, short_lengths, long_lengths, events

def sample_meal(theta, x0, init_state='F'):
    theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8 = theta

    theta1 = np.power(10., theta1)
    theta2 = np.power(10., theta2)
    theta3 = np.power(10., theta3)
    theta6 = np.power(10., theta6)
    theta7 = np.power(10., theta7)
    theta8 = np.power(10., theta8)

    x = np.array([x0])
    amount_eaten = 0.0
    k2 = 0.00055 # Digestion rate parameter
    t = 0.0
    state = init_state

    ## Use if starting in pause state
    f_length = []
    g_start = x0
    rate = theta3

    num_events = 0
    bout_lengths = []
    short_lengths = []
    long_lengths = []
    events = []

    while state != 'L':
        ## Feeding code
        if state == 'F':
            ## Get duration and rate
            #t_plus = np.round(np.random.exponential(1./theta1))
            rate = np.random.normal(theta2, theta3)
            while rate < 0: # ensure rate is positive
                rate = np.random.normal(theta2, theta3)

            t_plus = np.random.exponential(1./theta1)

            ## Integrate forward
            g_start = x[-1]
            f_length = np.linspace(0, t_plus, t_plus) # integrate at 1-second resolution
            x_segment = f_length*rate*3.5

            ## Append results, increment t
            x = np.hstack((x, x_segment))
            t += t_plus
            amount_eaten += t_plus*rate
            num_events += 1
            bout_lengths.append(t_plus)

            ## Find next state through transition kernel
            Q_val = ll.Q(x[-1], theta4, theta5)
            u = np.random.uniform()

            if Q_val < u:
                state = 'S'

            else:
                state = 'L'

        ## Short pause code
        elif state == 'S':
            ## Get duration
            t_plus = np.random.exponential(1./theta6)
            t_plus = int(t_plus)

            ## Integrate forward
            g_end_feeding = x[-1]
            t_c = int(2.*np.sqrt(g_end_feeding)/k2)
            p_length = np.arange(t_plus)
            x_segment = 0.25*np.power(k2*p_length, 2) - np.sqrt(g_end_feeding)*k2*p_length + g_end_feeding
            x_segment = np.clip(x_segment, 0, None)

            if t_plus > t_c:
                x_segment[t_c:] = 0.

            ## Append results, increment t
            x = np.hstack((x, x_segment))
            t += t_plus
            short_lengths.append(t_plus)

            event = [len(f_length), g_start, rate, len(p_length), g_end_feeding]
            events.append(event)

            ## Switch back to feeding
            state = 'F'

        ## Long pause code
        elif state == 'L':
            ## Get duration
            t_plus = ll.sample_L(x[-1], k2, theta7, theta8)
            t_plus = int(t_plus)

            ## Integrate forward
            g_end_feeding = x[-1]
            t_c = int(2.*np.sqrt(g_end_feeding)/k2)

            p_length = np.arange(t_plus)
            x_segment = 0.25*np.power(k2*p_length, 2) - np.sqrt(g_end_feeding)*k2*p_length + g_end_feeding
            x_segment = np.clip(x_segment, 0, None)

            if t_plus > t_c:
                x_segment[t_c:] = 0.

            ## Append results, increment t
            x = np.hstack((x, x_segment))
            t += t_plus
            long_lengths.append(t_plus)

            event = [len(f_length), g_start, rate, len(p_length), g_end_feeding]
            events.append(event)

            ## Switch back to feeding
            state = 'F'

        else:
            print "ERROR, invalid state"

    return x, amount_eaten, num_events, bout_lengths, short_lengths, long_lengths, events

def sample_lim_x(tmax, theta, x0, xmax, init_state='F'):
    theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8 = theta

    theta1 = np.power(10., theta1)
    theta2 = np.power(10., theta2)
    theta3 = np.power(10., theta3)
    theta6 = np.power(10., theta6)
    theta7 = np.power(10., theta7)
    theta8 = np.power(10., theta8)

    x = np.array([x0])
    amount_eaten = 0.0
    k2 = 0.00055 # Digestion rate parameter
    t = 0.0
    state = init_state

    ## Use if starting in pause state
    f_length = []
    g_start = x0
    rate = theta3

    num_events = 0
    bout_lengths = []
    short_lengths = []
    long_lengths = []
    events = []
    while t < tmax or state == 'L' or state == 'S':
        #print t
        ## Feeding code
        if state == 'F':
            ## Get duration and rate
            rate = np.random.normal(theta2, theta3)
            while rate < 0: # ensure rate is positive
                rate = np.random.normal(theta2, theta3)

            if rate <= 0:
                print rate

            t_plus = np.random.exponential(1./theta1)
            t_plus = np.ceil(t_plus)

            ## Integrate forward
            g_start = x[-1]

            predicted_x = g_start + t_plus*3.5*rate

            if predicted_x > xmax:
                t_plus = (xmax - g_start)/(3.5*rate)
                t_plus = np.floor(t_plus)

            f_length = np.arange(t_plus) + 1
            x_segment = g_start + 3.5*rate*f_length

            ## Append results, increment t
            x = np.hstack((x, x_segment))
            t += t_plus
            amount_eaten += t_plus*rate
            num_events += 1
            bout_lengths.append(t_plus)

            ## Find next state through transition kernel
            Q_val = ll.Q(x[-1], theta4, theta5)
            u = np.random.uniform()

            if Q_val < u:
                state = 'S'

            else:
                state = 'L'

        ## Short pause code
        elif state == 'S':
            ## Get duration
            t_plus = np.random.exponential(1./theta6)
            t_plus = int(t_plus)

            ## Integrate forward
            g_end_feeding = x[-1]
            t_c = int(2.*np.sqrt(g_end_feeding)/k2)
            p_length = np.arange(t_plus)
            x_segment = 0.25*np.power(k2*p_length, 2) - np.sqrt(g_end_feeding)*k2*p_length + g_end_feeding
            x_segment = np.clip(x_segment, 0, None)

            if t_plus > t_c:
                x_segment[t_c:] = 0.

            ## Append results, increment t
            x = np.hstack((x, x_segment))
            t += t_plus
            short_lengths.append(t_plus)

            event = [len(f_length), g_start, rate, len(p_length), g_end_feeding]
            events.append(event)

            ## Switch back to feeding
            state = 'F'

        ## Long pause code
        elif state == 'L':
            ## Get duration
            t_plus = ll.sample_L(x[-1], k2, theta7, theta8)
            t_plus = int(t_plus)

            ## Integrate forward
            g_end_feeding = x[-1]
            t_c = int(2.*np.sqrt(g_end_feeding)/k2)

            p_length = np.arange(t_plus)
            x_segment = 0.25*np.power(k2*p_length, 2) - np.sqrt(g_end_feeding)*k2*p_length + g_end_feeding
            x_segment = np.clip(x_segment, 0, None)

            if t_plus > t_c:
                x_segment[t_c:] = 0.

            ## Append results, increment t
            x = np.hstack((x, x_segment))
            t += t_plus
            long_lengths.append(t_plus)

            event = [len(f_length), g_start, rate, len(p_length), g_end_feeding]
            events.append(event)

            ## Switch back to feeding
            state = 'F'

        else:
            print "ERROR, invalid state"

    return x, amount_eaten, num_events, bout_lengths, short_lengths, long_lengths, events

def sample_refractory(tmax, theta, x0, period, init_state='F'):
    theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8 = theta

    theta1 = np.power(10., theta1)
    theta2 = np.power(10., theta2)
    theta3 = np.power(10., theta3)
    theta6 = np.power(10., theta6)
    theta7 = np.power(10., theta7)
    theta8 = np.power(10., theta8)

    x = np.array([x0])
    amount_eaten = 0.0
    k2 = 0.00055 # Digestion rate parameter
    t = 0.0
    state = init_state

    ## Use if starting in pause state
    f_length = []
    g_start = x0
    rate = theta3

    num_events = 0
    bout_lengths = []
    short_lengths = []
    long_lengths = []
    events = []
    while t < tmax or state == 'L' or state == 'S':
        #print t
        ## Feeding code
        if state == 'F':
            ## Get duration and rate
            #t_plus = np.round(np.random.exponential(1./theta1))
            rate = np.random.normal(theta2, theta3)
            while rate < 0: # ensure rate is positive
                rate = np.random.normal(theta2, theta3)

            if rate <= 0:
                print rate

            t_plus = np.random.exponential(1./theta1)
            t_plus = np.ceil(t_plus)

            ## Integrate forward
            g_start = x[-1]
            f_length = np.arange(t_plus) + 1
            x_segment = g_start + 3.5*rate*f_length

            ## Append results, increment t
            x = np.hstack((x, x_segment))
            t += t_plus
            amount_eaten += t_plus*rate
            num_events += 1
            bout_lengths.append(t_plus)

            ## Find next state through transition kernel
            Q_val = ll.Q(x[-1], theta4, theta5)
            u = np.random.uniform()

            if Q_val < u:
                state = 'S'

            else:
                state = 'L'

        ## Short pause code
        elif state == 'S':
            ## Get duration
            t_plus = np.random.exponential(1./theta6)
            t_plus = int(t_plus)

            ## Integrate forward
            g_end_feeding = x[-1]
            t_c = int(2.*np.sqrt(g_end_feeding)/k2)
            p_length = np.arange(t_plus)
            x_segment = 0.25*np.power(k2*p_length, 2) - np.sqrt(g_end_feeding)*k2*p_length + g_end_feeding
            x_segment = np.clip(x_segment, 0, None)

            if t_plus > t_c:
                x_segment[t_c:] = 0.

            ## Append results, increment t
            x = np.hstack((x, x_segment))
            t += t_plus
            short_lengths.append(t_plus)

            event = [len(f_length), g_start, rate, len(p_length), g_end_feeding]
            events.append(event)

            ## Switch back to feeding
            state = 'F'

        ## Long pause code
        elif state == 'L':
            ## Get duration
            t_plus = ll.sample_L(x[-1], k2, theta7, theta8)
            t_plus = int(t_plus)

            ## Ensure time is greater than refractory period
            if t_plus < period:
                t_plus = period

            ## Integrate forward
            g_end_feeding = x[-1]
            t_c = int(2.*np.sqrt(g_end_feeding)/k2)

            p_length = np.arange(t_plus)
            x_segment = 0.25*np.power(k2*p_length, 2) - np.sqrt(g_end_feeding)*k2*p_length + g_end_feeding
            x_segment = np.clip(x_segment, 0, None)

            if t_plus > t_c:
                x_segment[t_c:] = 0.

            ## Append results, increment t
            x = np.hstack((x, x_segment))
            t += t_plus
            long_lengths.append(t_plus)

            event = [len(f_length), g_start, rate, len(p_length), g_end_feeding]
            events.append(event)

            ## Switch back to feeding
            state = 'F'

        else:
            print "ERROR, invalid state"

    return x, amount_eaten, num_events, bout_lengths, short_lengths, long_lengths, events
