from collections import defaultdict
import numpy as np

# reads text and stores data in header to list mapping
def read_text(fname, headers, sep=' '):
    header_to_data = defaultdict(list)
    with open(fname) as f:
        for line in f:
            line = line.strip()
            cols = line.split(sep)
            for i in range(len(cols)):
                if len(cols) == 7:
                    header_to_data[headers[i]].append(cols[i])
    return header_to_data


#TODO: Add the B/M data.
# Add NORMS
# reads text and stores data in header to list mapping
def read_X_Y(fname, headers, sep=' '):
    header_to_data = defaultdict(list)
    Y = []
    X = []
    Y_names = []

    ben_mag_data = [] # B, M, -1 (for N/A)
    with open(fname) as f:
        for line in f:
            line = line.strip()
            cols = line.split(sep)

            x_name = 'all-mias/' + cols[0] + '.pgm'
            # if x_name in X:
            	#remove x_name from X
            if len(cols) == 7 and not cols[2] == 'NORM':
                X.append('all-mias/' + cols[0] + '.pgm')
                Yi = np.array( [float(cols[4]),
                             float(cols[5]),
                            float(cols[6])] ).reshape((3,1))
                Y_names.append(cols[0])
                Y.append(Yi)
                ben_mag_data.append(cols[1])
            #need to deal with the NORMS:
            if len(cols) == 3 and cols[2] == 'NORM':
                X.append('all-mias/' + cols[0] + '.pgm')
                Yi = np.array( [0.0,0.0,0.0] ).reshape((3,1))
                Y_names.append(cols[0])
                Y.append(Yi)
                ben_mag_data.append('-1')

    return  X,np.array(Y), Y_names, ben_mag_data



# returns a numpy array of the data (column) requested
def get_data_col(header_to_data, header, cast_int=False, cast_float=False):
    data = header_to_data[header]
    if cast_int:
        data = [int(datum) for datum in data]
    elif cast_float:
        data = [float(datum) for datum in data]
    return np.array(data)

def filter_data_by_colval(header_to_data,col, col_val):
    filter_fnames = [ 'all-mias/' + header_to_data['file_name'][index] + '.pgm'  
            for index,val in enumerate(header_to_data[col]) if val != col_val]
    Y = [np.array([header_to_data['x'][index],header_to_data['y'][index], header_to_data['radius'][index]]).reshape((3,1)) 
            for index,val in enumerate(header_to_data[col]) if val != col_val]
    return filter_fnames, Y
