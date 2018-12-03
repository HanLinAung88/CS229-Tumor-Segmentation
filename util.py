from collections import defaultdict
import numpy as np
import pydicom
import os
import numpy as np
import re
import cv2

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
    Y_names = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            cols = line.split(sep)
            if len(cols) == 7:
                X.append('all-mias/' + cols[0] + '.pgm')
                Yi = np.array( [float(cols[4]),
                             float(cols[5]),
                            float(cols[6])] ).reshape((3,1))
                Y_names.append(cols[0])
                Y.append(Yi)
    return  X,np.array(Y), Y_names



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

#gets X and Y dicom data, and X_benign of all images given the directory (this is the *function* to call)
def get_X_Y_dicom(directory):
    orig_im_map, mask_im_map = read_dicomIm_dir(directory)
    X = []
    X_benign = []
    Y = []
    for im in orig_im_map:
        if im in mask_im_map:
            X.append(orig_im_map[im])
            Y.append(mask_im_map[im])
        else:
            X_benign.append(im)
    return np.array(X), np.array(Y), np.array(X_benign)

#resizes image into a certain scale
def resize_image(im, x_size=1024, y_size=1024):
    resized_image = cv2.resize(im, (x_size, y_size))
    return resized_image

# extracts the patient id
def extract_patient_id(name):
    patient_id = re.search('_P_(.*?)_', name)
    if patient_id.group() != None:
        return patient_id.group(1)
    return None

#reads directory for all dicom images
def read_dicomIm_dir(directory, resize=True, x_size=1024, y_size=1024):
    orig_im_map = {}
    mask_im_map = {}
    for subdir, dirs, files in os.walk(directory):
        for curr_file in files:
            if curr_file.endswith('.dcm'):
                patient_id = extract_patient_id(subdir)
                dicom_im = read_dicomIm(os.path.join(subdir, curr_file))
                if resize:
                    dicom_im = resize_image(dicom_im, x_size, y_size)
                if patient_id is not None:
                    #_1 is mask folder 000000.dcm is zoomed in image of tumor
                    if '_1/' in subdir:
                        if curr_file != '000000.dcm':
                            mask_im_map[patient_id] = dicom_im
                    else:
                        orig_im_map[patient_id] = dicom_im
    return orig_im_map, mask_im_map

#reads dicom image and converts it to be between 0 and 255
def read_dicomIm(filename):
    ds = pydicom.dcmread(filename)
    # https://github.com/pydicom/pydicom/issues/352
    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)

    # Rescaling data between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)
    return image_2d_scaled
