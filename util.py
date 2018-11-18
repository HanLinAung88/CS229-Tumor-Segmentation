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
                header_to_data[headers[i]].append(cols[i])
    return header_to_data

# returns a numpy array of the data (column) requested
def get_data_col(header_to_data, header, cast_int=False, cast_float=False):
    data = header_to_data[header]
    if cast_int:
        data = [int(datum) for datum in data]
    elif cast_float:
        data = [float(datum) for datum in data]
    return np.array(data)
