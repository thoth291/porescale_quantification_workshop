import tifffile
import matplotlib.pyplot as plt
import skimage
from src import utils
from src.analysis import ImageQuantifier

if __name__ == "__main__":
    # Download the data files
    # utils.get_datafiles()

    # Check a slice of the image
    img = ImageQuantifier("data/beadpack.tif")
    img.plot_slice()
    img.run_analysis(heterogeneity_kwargs={'no_radii': 20, 'no_samples_per_radius': 500}, ev_kwargs={'cube_size': 256})
