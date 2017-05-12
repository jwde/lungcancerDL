import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
from joblib import Parallel, delayed
import multiprocessing
import argparse
from skimage import measure, morphology


def get_dicoms(input_dir):
    patients = os.listdir(input_dir)
    patients.sort()
    npat = len(patients)
    # for some odd reason there are duplicate files for each folder. Each duplicate starts with '._'
    for i in range(npat):
        if '._' in patients[i]:
            patients[i] = []
    patients = list(filter(None,patients))
    return patients

# Load the scans in given folder path
def load_scan(path):
    # edited to skip all files starting with '._'
    # Also, file is missing 'DICM' marker. Use force=True to force reading
    slices = [dicom.read_file(path + '/' + s, force=True) for s in os.listdir(path) if '._' not in s]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    # Modification by Bo: Some rare slices are smaller than 512*512. Delete these slices
    # Make sure the slice thickness of the slice before the deletion is twice as thick
    rind = []
    for ii in range(len(slices)):
        if len(slices[ii].PixelData)==2*slices[ii].Rows**2:
            slices[ii].SliceThickness = slice_thickness
        else:
            rind.append(ii) # save index of slice to be removed
    
    if len(rind)==1:
        slices[rind[0]-1].SliceThickness = 2*slice_thickness
        del slices[rind[0]]
    elif len(rind)>1:
        for ii in sorted(rind,reverse=True): # if multiple bad slices delete in reverse
            slices[ii-1].SliceThickness = 2*slice_thickness
            del slices[ii]
    
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

# Trim excess 0s (to make mask smaller and higher res)
def trim_excess(mask, image):
    # This function assumes that z-axis (height) is the 1st coordinate
    height = mask.shape[0]
    width = mask.shape[1]
    # Use mask to determined edges of the trim
    ind = 0
    while np.sum(mask[ind,:,:]) == 0:
        ind += 1
    bottom = ind
    
    ind = 0
    
    while np.sum(mask[height-1-ind,:,:]) == 0:
        ind += 1
    top = ind
    
    ind = 0
    while np.sum(mask[:,ind,:]) == 0:
        ind += 1
    front = ind
    
    ind = 0
    while np.sum(mask[:,width-1-ind,:]) == 0:
        ind += 1
    back = ind
    
    ind = 0
    while np.sum(mask[:,:,ind]) == 0:
        ind += 1
    left = ind
    
    ind = 0
    while np.sum(mask[:,:,width-1-ind]) == 0:
        ind += 1
    right = ind
    
    #trimmed_mask = mask[bottom:(height-top),front:(width-back),left:(width-right)]
    # I don't need trimmed mask anymore
    trimmed_image = image[bottom:(height-top),front:(width-back),left:(width-right)]
    
    return trimmed_image

# 2-stage resampling
def resample(image, scan, new_spacing=[1,1,1], final_shape=[25,50,60]):
    # This function modified by Bo for 2-stage resampling
    # 1st stage: make pixel spacing equal in all directions
    # 2nd stage: downsample image shape
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))
    
    # Some scans do not have slice thickness!!!
    # If this is the case, estimate the thickness based on number of scans
    if spacing[0] == 0:
        if len(scan) < 145:
            spacing[0] = 2.5
        elif len(scan) > 190:
            spacing[0] = 1.5
        else:
            spacing[0] = 2

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='mirror')
    
    final_resize_factor = final_shape / new_shape
    image = scipy.ndimage.interpolation.zoom(image, final_resize_factor, mode='mirror')
    
    return image

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask1(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image
#======================= Preprocess Images ==========================#

def main_loop(INPUT_FOLDER,OUT_FOLDER,patients,i,trim):
    patient = patients[i]
    npat = len(patients)
    patientINFO = load_scan(INPUT_FOLDER + patient)
    print('Preprocessing '+patient+' || '+str(npat-i-1)+' remain')
    patient_pixels = get_pixels_hu(patientINFO)
    # Segmentation with fill (for trimming)
    segmented_lungs_fill = segment_lung_mask1(patient_pixels, True)
    # Dilate segmentation to create mask
    kernel = np.ones((5,5),np.uint8)
    dilated_mask = scipy.ndimage.morphology.binary_dilation(segmented_lungs_fill, iterations=5)
    # mask image
    masked_image = patient_pixels
    masked_image[dilated_mask==0] = -10000 # mask the image by logical indexing
    # trim excess around ROI
    trimmed_pixels = trim_excess(dilated_mask, masked_image) if trim else masked_image

    # 2-stage resampling (equal spacing, lower resolution)
    pix_resampled = resample(trimmed_pixels, patientINFO,[2,2,2], [60,227,227])
    
    outfile = os.path.join(OUT_FOLDER, '%s.npy' % patient)
    np.save(outfile, pix_resampled)
if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT_FOLDER', help="Path to input data directory")
    parser.add_argument('OUT_FOLDER', help="Path to output data directory")
    parser.add_argument('--trim', '-t', action=store_true, default=False help="attempt to trim black border around lung")
    r = parser.parse_args()
    if not os.path.exists(r.OUT_FOLDER):
        os.makedirs(r.OUT_FOLDER)
    dicoms = get_dicoms(r.INPUT_FOLDER)

    print ("There are " + str(len(dicoms)) + " dicom files to process.")
    print ("Running " + str(num_cores) + " ways parallel.")
    Parallel(n_jobs=num_cores)(delayed(main_loop)(r.INPUT_FOLDER,r.OUT_FOLDER,dicoms,i, r.trim) for i in range(len(dicoms)))

