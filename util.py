from collections import defaultdict
from glob import glob
from matplotlib.pyplot import imread, imshow
import matplotlib.pyplot as plt
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
    patient_id = re.search('_P_(.*?)_(.*?)_(.*?)(?:_|\/)', name)
    if patient_id.group() != None:
        patient_id_group = patient_id.group(1) + patient_id.group(2) + patient_id.group(3)
        return patient_id_group
    return None

#reads directory for all dicom images
def read_dicomIm_dir(directory, resize=True, x_size=1024, y_size=1024):
    orig_im_map = {}
    mask_im_map = {}
    MAX_ZOOMEDIN_FILESZ = 2000000
    for subdir, dirs, files in os.walk(directory):
        for curr_file in files:
            if curr_file.endswith('.dcm'):
                patient_id = extract_patient_id(subdir)
                dicom_im = read_dicomIm(os.path.join(subdir, curr_file))
                if resize:
                    dicom_im = resize_image(dicom_im, x_size, y_size)
                if patient_id is not None:
                    #ROI: Region of image
                    if 'ROI' in subdir:
                        #if patient_id is already there, then concat image with multiple masks
                        #file size of actual mask image is bigger than zoomed in image
                        if os.stat(os.path.join(subdir, curr_file)).st_size >= MAX_ZOOMEDIN_FILESZ:
                            if patient_id in mask_im_map:
                                for row in range(mask_im_map[patient_id].shape[0]):
                                    for col in range(mask_im_map[patient_id].shape[1]):
                                        if dicom_im[row][col] == 255:
                                            mask_im_map[patient_id][row][col] = 255
                            else:
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

#extracts features for logistic regression given input and output mask
#X: num_images * input_width * input_height (input image)
#Y: num_images * input_width * input_height (mask)
def extract_logReg_data(X, Y):
    #converts Y to 1 and 0, 1 for tumor and 0 for non-tumor
    Y[Y == 255] = 1
    Y[Y != 1] = 0
    flattend_size = X.shape[0] * X.shape[1] * X.shape[2]
    return X.reshape(flattend_size, 1), Y.flatten()

#the Y mask function to produce masks as in the milestone jupyter notebook
def produce_y_mask(Y,Y_names):
    for i in range(len(Y)):
        image = np.zeros((1024,1024,1))
        cv2.circle(image, (int(Y[i][0]),1024-int(Y[i][1])),int(Y[i][2]),(255),-1)
        name = 'mias_y_masked/' + Y_names[i] + '.png'
        cv2.imwrite(name, image)

#extracts data
def extract_data_CBIS_MIAS(isCBIS=True, isMias=True):
    X = None
    Y = None
    if(isCBIS):
        X, Y, X_benign = get_X_Y_dicom('../ddsm/CBIS-DDSM')
    if(isMias):
        file_names = sorted(glob('all-mias/*.pgm'))
        headers = ['file_name', 'character', 'class', 'severity', 'x', 'y', 'radius']
        X_filtered, Y_headers,Y_names = read_X_Y('dataset/data.txt',headers)
        X_mias = np.array([imread(file_name) for file_name in X_filtered])
        produce_y_mask(Y_headers,Y_names)
        Y_mask_filenames = ['mias_y_masked/' + file_name + '.png' for file_name in Y_names]
        Y_mias = np.array([cv2.imread(file_name, cv2.IMREAD_GRAYSCALE) for file_name in Y_mask_filenames])
        if X is not None:
            X = np.concatenate((X, X_mias), axis=0)
            Y = np.concatenate((Y, Y_mias), axis=0)
        else:
            X = X_mias
            Y = Y_mias
    return X, Y

#converts image in-place (0 to for non-tumor, 1 for tumor) to binary format
#assumes values are either 255 or 0 for it to work properly
def convert_im_to_binary(Y):
    Y[Y == 255] = 1
    Y[Y != 1] = 0
    return Y
