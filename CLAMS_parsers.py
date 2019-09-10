import numpy as np
import csv
from datetime import datetime, timedelta

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

    if mass == '':
        mass = headers[7][0].split(':')[1]

    mass = mass.split(' grams')[0]
    
    data = []
    for count, i in enumerate(preproc_data[4:-1]):
        ## deals with some places where extra working has been done inside the csv
        ## which creates extra non-data lines
        if i[3] == '':
            continue
        else:
            start = datetime.strptime((i[3] + ' ' + i[4]).strip(' '), '%d/%m/%Y %H:%M:%S')
            end = datetime.strptime((i[3] + ' ' + i[5]).strip(' '), '%d/%m/%Y %H:%M:%S')
            amount_eaten = float(i[7])

        if start.hour > end.hour: # the bout has spanned midnight
            end = end + timedelta(days=1)
            
        data.append((start, end, amount_eaten))

    return num_id, subject_id, mass, data