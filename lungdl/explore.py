import numpy as np
import pandas as pd
#import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import argparse

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    #p = image.transpose(2,1,0)
    p = image
    
    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def lung_to_path(dir, lung_id):
    return dir + lung_id + ".npy"
def load_lung(dir,lung_id):
    return np.load(lung_to_path(dir,lung_id))
def get_random_lung(dir,lung_files):
    random_lung_id = np.random.choice(lung_files, 1)[0]
    return load_lung(dir,random_lung_id)

def numpy_plot_rand(input_dir):
    lung_files = [os.path.splitext(f)[0] for f in os.listdir(input_dir)]
    print (input_dir, lung_files[:10])
    lung = get_random_lung(input_dir,lung_files)
    print (lung.shape)
    plt.imshow(lung[30,:,:], cmap=plt.cm.gray)
    plt.show()

def plot(img_path):
    lung = np.load(path)
    plt.imshow(lung[30,:,:], cmap=plt.cm.gray)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT_FOLDER', help="Path to input data directory")
    r = parser.parse_args()
    patients = os.listdir(r.INPUT_FOLDER)
    patients.sort()
    numpy_plot_rand(r.INPUT_FOLDER)


