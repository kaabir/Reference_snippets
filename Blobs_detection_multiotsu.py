import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.filters import threshold_multiotsu
from scipy.signal import find_peaks
import napari_simpleitk_image_processing as nsitk
from skimage.io import imread, imshow
from skimage.measure import label, regionprops

# Get voxel of pixel intensity values inside the mask 
def replace_intensity(mask, img):
    if not (mask.shape == img.shape):
        return False
    
    mat_intensity = np.where(np.logical_and(mask,img),img,0) # Overlap Mask on original image (as reference)
    return mat_intensity

image_or = imread('image.tif')
   
image_dn = nsitk.median_filter(image_or, radius_x=3, radius_y=3, radius_z=0)
threshold_initial = nsitk.threshold_otsu(image_dn) 
image = replace_intensity(threshold_initial, image_or)
image_dn = nsitk.median_filter(image, radius_x=3, radius_y=3, radius_z=0)
# Apply multi-Otsu thresholding 
thresholds = threshold_multiotsu(image_dn)#, classes=num_classes)
# The class 1 is thresholds[0] == mean and thresholds[1] is class 2 histogram above mean
blobs = image_dn > thresholds[0]+thresholds[1]
# Use the threshold values to segment the image
# regions = np.digitize(image_dn, bins=thresholds)

# # Get histogram counts
# hist, bins = np.histogram(image_dn.ravel(), bins=255)

# # Find peaks in the histogram
# peaks, _ = find_peaks(hist)

# # Filter peaks to only include those greater than the second class threshold
# filtered_peaks = [peak for peak in peaks if bins[peak] > second_class_threshold]

# Plot the results
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Original image
ax[0].imshow(image_or, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

# # Histogram with thresholds and peaks
# ax[1].hist(image_dn.ravel(), bins=255, color='gray', edgecolor='black')
# ax[1].set_title('Histogram with thresholds')
# for thresh in thresholds:
#     ax[1].axvline(thresh, color='r', linestyle='dashed')
# for peak in filtered_peaks:
#     ax[1].axvline(bins[peak], color='g', linestyle='dashed')

# Multi-Otsu result
ax[1].imshow(blobs, cmap='gray')
ax[1].set_title('Multi-Otsu Result')
ax[1].axis('off')

plt.show()

# Output peak information
# print("Histogram Peaks:")
# for peak in peaks:
#     print(f"Intensity: {bins[peak]}, Count: {hist[peak]}")
