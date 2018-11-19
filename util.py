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


# reads text and stores data in header to list mapping
def read_X_Y(fname, headers, sep=' '):
    header_to_data = defaultdict(list)
    Y = []
    X = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            cols = line.split(sep)
            if len(cols) == 7:
                X.append('all-mias/' + cols[0] + '.pgm')
                Yi = np.array( [float(cols[4]),
                             float(cols[5]),
                            float(cols[6])] ).reshape((3,1))
                Y.append(Yi)
    return  X,np.array(Y)



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
