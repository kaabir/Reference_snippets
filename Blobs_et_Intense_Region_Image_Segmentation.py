import numpy as np
from skimage import filters, draw, exposure, feature, measure, morphology, img_as_float32, img_as_ubyte
from skimage.morphology import remove_small_objects
import pyclesperanto_prototype as cle
import napari_simpleitk_image_processing as nsitk
from scipy.ndimage import label
from skimage.filters import threshold_multiotsu
#from skimage.measure import shannon_entropy
from math import sqrt, pi
import matplotlib.pyplot as plt

def extract_blobs(image, min_sigma=None, max_sigma=None, num_sigma=None, threshold=None):
    """
    
    Parameters
    ----------
    image : Array of 8/16bit
    
    min_sigma : TYPE, optional
        For detecting small blobs the value is 2.
    max_sigma : TYPE, optional
        Reduced max_sigma for smaller blobs. The default is 10.
    num_sigma : TYPE, optional
        Reduced number of scales. The default is 10.
    threshold : TYPE, optional
        Increased threshold for prominent blobs. The default is around 0.1

    Returns
    -------
    Binary Array of regions intense regions in form of blobs.

    """
    # Top-hat filter and denoising
    image_thf = cle.top_hat_sphere(image, None, 10.0, 10.0, 0.0)
    image_dn = nsitk.median_filter(image_thf, radius_x=2, radius_y=2, radius_z=0)
    image_dn2 = nsitk.median_filter(image, radius_x=2, radius_y=2, radius_z=0)
    
    try:
        # Multi-Otsu thresholding
        image_motsu = threshold_multiotsu(image_dn, classes=3)
        image_segM = image_dn > (image_motsu[0] + image_motsu[1])
        image_segF = remove_small_objects(image_segM, min_size=10)

        # Blob detection
        # Calculate max_sigma for a maximum blob area of 200 pixels squared
        # max_area = 200
        # max_radius = sqrt(max_area / pi)
        # max_sigma = max_radius / sqrt(2)
        blobs_log = feature.blob_log(
                        image_dn2, 
                        min_sigma=min_sigma, 
                        max_sigma=max_sigma,  # Reduced max_sigma for smaller blobs
                        num_sigma=num_sigma,  # Reduced number of scales
                        threshold=threshold  # Increased threshold for prominent blobs
                        )
        
        
        # blobs_log  = feature.blob_doh(image_dn, min_sigma=3, max_sigma=5, num_sigma=5, threshold=0.001)
        blobs_log[:, 2] *= sqrt(2)
    
        # Create blob mask
        blob_mask = np.zeros_like(image, dtype=int)
        for blob in blobs_log:
            y, x, r = blob.astype(int)
            rr, cc = draw.disk((y, x), r, shape=image.shape)
            blob_mask[rr, cc] = 1
        
        # Combine masks and label
        speckles = (blob_mask & image_segF).astype(int)

        # Visualization (consider moving this to a separate function if not always needed)
        # fig, ax = plt.subplots(1, 4, figsize=(18, 6))
        # ax[0].imshow(image, cmap='gray')
        # ax[0].set_title('Original Image')
        # ax[1].imshow(image_segF, cmap='gray')
        # ax[1].set_title('Multi OTSU Thresholding')
        # ax[2].imshow(blob_mask, cmap='gray')
        # ax[3].set_title('Blobs Mask')
        # ax[3].imshow(speckles, cmap='gray')
        # ax[2].set_title('Extracted Blobs')
        # for a in ax:
        #     a.axis('off')
        # plt.tight_layout()
        # plt.show()

        return speckles
    
    except ValueError as e:
        print(e)
        return measure.label(np.zeros_like(image, dtype=np.uint8))
    
extracted_blobs = extract_blobs(img, min_sigma=4, max_sigma=10, num_sigma=20, threshold=0.025) 

def extract_multiotsu(image):
    """

    Parameters
    ----------
    image : Array of 8/16bit

    Returns
    -------
    TYPE
        Binary Array of regions above mean intensity of all morphologies.

    """
    # Top-hat filter and denoising
    image_thf = cle.top_hat_sphere(image, None, 10.0, 10.0, 0.0)
    image_dn = nsitk.median_filter(image_thf, radius_x=2, radius_y=2, radius_z=0)

    try:
        # Multi-Otsu thresholding
        image_motsu = threshold_multiotsu(image_dn, classes=3)
        image_segM = image_dn > (image_motsu[0] + image_motsu[1])
        image_segM1 = image_dn > image_motsu[1]
        image_segF = remove_small_objects(image_segM, min_size=10)

        # Visualization (consider moving this to a separate function if not always needed)
        fig, ax = plt.subplots(1, 4, figsize=(18, 6))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original Image')
        ax[1].imshow(image_segM, cmap='gray')
        ax[1].set_title('Multi OTSU Thresholding')
        ax[2].imshow(image_segF, cmap='gray')
        ax[2].set_title('Filtered Thresholding')
        ax[3].imshow(image_segM1, cmap='gray')
        ax[3].set_title('Multi OTSU 1 Thresholding')        
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.show()

        return image_segF
    
    except ValueError as e:
        print(e)
        return measure.label(np.zeros_like(image, dtype=np.uint8))