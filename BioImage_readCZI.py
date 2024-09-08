from bioio import BioImage
import bioio_czi
file = "ABCD.czi"
# img = BioImage("my_file.tiff")  # selects the first scene found
img = BioImage(file, reader=bioio_czi.Reader)
img.data # returns 5D TCZYX numpy array
# img.xarray_data  # returns 5D TCZYX xarray data array backed by numpy
# img.dims  # returns a Dimensions object
# img.dims.order  # returns string "TCZYX"
# img.dims.X
print(img.dims.order)  # returns size of X dimension
# img.shape  # returns tuple of dimension sizes in TCZYX order
# img.get_image_data("CZYX", T=0)  # returns 4D CZYX numpy array

# Get physical size in micrometers for each dimension
physical_size_x = img.physical_pixel_sizes.X
physical_size_y = img.physical_pixel_sizes.Y
physical_size_z = img.physical_pixel_sizes.Z

print(f"Physical size (μm): X: {physical_size_x}, Y: {physical_size_y}, Z: {physical_size_z}")