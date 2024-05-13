# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:16:31 2024

Workflow
Open image (If 3D - get Z-stack)
Smooth the image using gaussian blur (sigma=2) or Remove background Noise
Threshold to get mask 
Post process - Fill holes, removed edges, small labels, filter etc 
haesleinhuepf.github.io/BioImageAnalysisNotebooks/25_neighborhood_relationships_between_cells/05_count_touching_neighbors.html

"""

import pyclesperanto_prototype as cle
import numpy as np
import pandas as pd

# Example image to test
cells = cle.artificial_tissue_2d(
    delta_x=48, 
    delta_y=32, 
    random_sigma_x=7, 
    random_sigma_y=7, 
    width=250, 
    height=250)

cle.imshow(cells, labels=True)

# Mesh between neighboring cells
mesh = cle.draw_mesh_between_touching_labels(cells)

cle.imshow(mesh)
# Centroid connections and cell borders
visualization = mesh * 2 + cle.detect_label_edges(cells)

cle.imshow(visualization, color_map='jet')

# Analyze and visualize number of touching neighbors
neighbor_count_image = cle.touching_neighbor_count_map(cells)

cle.imshow(neighbor_count_image, color_map='jet', colorbar=True, min_display_intensity=0)

# Remove borders
cells_ex_border = cle.exclude_labels_on_edges(cells)

cle.imshow(cells_ex_border, labels=True)

# correct the parametric image
neighbor_count_image_ex_border = neighbor_count_image * (cells_ex_border != 0)

cle.imshow(neighbor_count_image_ex_border, color_map='jet', colorbar=True, min_display_intensity=0)


neighbor_count_image_ex_border = neighbor_count_image * (cells_ex_border != 0)

cle.imshow(neighbor_count_image_ex_border, color_map='jet', colorbar=True, min_display_intensity=0)

# Measure the number of neighbours
cle.read_intensities_from_map(cells_ex_border, neighbor_count_image_ex_border)

# Statistics
statistics = cle.statistics_of_labelled_pixels(neighbor_count_image_ex_border, cells_ex_border)

table = pd.DataFrame(statistics)

# rename a column
table = table.rename(columns={"mean_intensity": "number_of_neighbors"})

# only filter out a subset of all columns; only what we care
table = table[["label", "number_of_neighbors", "centroid_x", "centroid_y"]]


#

'''
import numpy as np
import aicsshparam
from aicsshparam import shparam, shtools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from aicsshparam import shtools, shparam
from skimage.morphology import ball, cube, octahedron
from skimage.io import imread, imshow
from scipy import ndimage

import napari_simpleitk_image_processing as nsitk
import pyclesperanto_prototype as cle
from skimage import img_as_float32
from skimage.util import img_as_ubyte
from tifffile import imwrite
from skimage import measure

def folder_scan(directory, extension, marker=None, Key=str()):
    # folder_scan(chromo_folder, ".xlsx", marker="(Chromo)")
    get_files = []
    extension = extension  # ".xlsx"  # ".czi"
    final_dir = directory + Key
    
    try:
        for f_name in os.listdir(final_dir):
            if marker is not None:
                if f_name.find(marker) != -1 and f_name.endswith(extension):
                    get_files.append(os.path.join(final_dir, f_name))
            else:
                if f_name.endswith(extension):
                    get_files.append(os.path.join(final_dir, f_name))
        return get_files              

    except FileNotFoundError:
        print(">>> This Folder Not Present", Key)
        print(final_dir)  

ctrl = ['E:/Quantified/test/Spherical Harmonics/fromMask/']

OutDir = 'E:/Quantified/test/Spherical Harmonics/fromMask/'
Result_folder = os.path.join(OutDir, 'Result')
if not os.path.exists(Result_folder):
    os.makedirs(Result_folder)

# Initialize an empty list to collect coefficients for each set
all_coeffs = []

# Define a function to process each set of paths
def process_paths(paths):
    coeffs_list = []
    for index, path in enumerate(paths):   
        count = 0
        for filename in os.listdir(path):
            if filename.endswith(".tif"):
                count += 1
                print(filename)
                img_path = os.path.join(path, filename)
            
            # Define the anisotropic pixel scaling factors
                Z_Scaling = 0.3054316 #0.0263576/0.3054316 #3*0.0263576  # z = 0.5um BINNED3*
                Y_Scaling = 0.0263576#0.0263576/0.0263576 #1.0  # Assuming Y and X pixels are already isotropic = 0.0263576
                X_Scaling = 0.0263576#0.0263576/0.0263576
                
                
            # Load your 3D volume with shape (ZXY = 20, 1024, 1024) into a numpy array called 'volume'

                #print(img_path)
                img = imread(img_path)  # Load image
                print("Img shape:", img.shape)
                lbl_nom = os.path.basename(os.path.dirname(path))  # Extract parent folder name as label
                print(lbl_nom)
                resampled = cle.scale(img, factor_x=X_Scaling, factor_y=Y_Scaling, factor_z=Z_Scaling, auto_size=True)
                print("Cle Img shape:", resampled.shape)
                original_img = img.shape
                Z_Scaling = original_img[0] * Z_Scaling
                    # Calculate the new shape with isotropic pixels
                new_img = (int(original_img[0] * Z_Scaling), original_img[1], original_img[2])
                print("Z Scaling Factor:",Z_Scaling)    
                # Rescale the volume along the Z axis using interpolation
                rescaled_img = ndimage.zoom(img, (Z_Scaling, X_Scaling, Y_Scaling), order=5, mode='nearest')
                    # interpolate along other axes as well (Y and X) to ensure isotropic voxels,
                    # Verify the new shape
                print("New shape:", rescaled_img.shape)
                    # # Donwsampled Image                   
                # nucelus_GB = nsitk.gaussian_blur(rescaled_img, 2.0, 2.0, 2.0)
                    # nucelus_GC = cle.gamma_correction(nucelus_GB, None, 1.0)
                # nucelus_OT = nsitk.threshold_otsu(nucelus_GB)              
                
                # label_OrgCnt = measure.label(nucelus_OT)
                # # Calculate the number of stacks for each label
                # label_counts = np.bincount(label_OrgCnt.ravel())         
                # print("Number of Unique mask Found", np.unique(label_OrgCnt)[1:])
                # for lbl_count in np.unique(label_OrgCnt)[1:]:
                                     
                #         #######
                #         # View Segmentation
                #         #######
                #         #viewer = napari.Viewer()         
                #         maskLBL = label_OrgCnt == lbl_count
                    # nucelus_CL = cle.closing_labels(nucelus_OT, None, 1.0)
                    # nucelus_BF = nsitk.binary_fill_holes(nucelus_CL)
                    # nucelus_FLH = fill_large_hole(nucelus_BF)
                                     
                # img_SH = img_as_float32(nucelus_OT, force_copy=False)     
                # os.chdir(Result_folder)
                # imwrite("(Mask)_"+filename +".tif", img_SH)              

                # nucleus_mask = nsitk.threshold_otsu(nucelus_GB)                     

                # Binning process
                bin_size = 2
                remainder = rescaled_img.shape[0] % bin_size
                if remainder != 0:
                    rescaled_img = rescaled_img[:-remainder, :, :]  # Trim to make divisible
                binned_img = np.mean(rescaled_img.reshape(rescaled_img.shape[0] // bin_size, bin_size,
                                          rescaled_img.shape[1] // bin_size, bin_size,
                                          rescaled_img.shape[2] // bin_size, bin_size),
                      axis=(1, 3, 5))
                    
                print("Binned shape:", binned_img.shape)

                    # Calculate the spherical harmonics expansion up to order lmax = 2
                (coeffs, grid_rec), (image_, mesh, grid, transform) = shparam.get_shcoeffs(image=binned_img, lmax=8)
                coeffs.update({'label': lbl_nom})  # Add label to the coefficients dictionary
                coeffs.update({'filename': filename})
                coeffs_list.append(coeffs)  # Append coefficients to the list
                pd.DataFrame(coeffs_list).to_excel('osmotic_shock_spherical_harmonics.xlsx')
                    
                    # Calculate the corresponding reconstruction error
                mse = shtools.get_reconstruction_error(grid,grid_rec)    
                print(mse)
                coeffs.update({'Error': mse})

                    # Reconstruct mesh from grid
                mesh_rec = aicsshparam.shtools.get_reconstruction_from_grid(grid_rec)

                    # Save mesh
                aicsshparam.shtools.save_polydata(mesh_rec, filename+'_'+ str(count)+'mesh_rec.vtk')
                # # Visualize the mesh
                aicsshparam.shtools.visualize_mesh(mesh)                
            
            # from mayavi import mlab

            # # Load VTK mesh file
            # mesh_file = filename+'_'+ str(count)+'mesh_rec.vtk'
            # mesh = mlab.pipeline.open(mesh_file)

            # # Visualize the mesh
            # mlab.pipeline.surface(mesh)
            # mlab.show()

            
    return coeffs_list

# Process each set of paths

ctrl_coeffs = process_paths(ctrl)

# Concatenate all coefficients DataFrames into one
# df_coeffs = pd.concat([pd.DataFrame(coeffs) for coeffs in (ctrl_coeffs, five_Um_coeffs, ten_Um_coeffs,twen_Um_coeffs)], ignore_index=True)
# os.chdir(Result_folder)
# pd.DataFrame(df_coeffs).to_excel('osmotic_shock_spherical_harmonics.xlsx')
'''