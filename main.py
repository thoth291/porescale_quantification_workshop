import tifffile
import matplotlib.pyplot as plt
import skimage
from src import utils

if __name__ == "__main__":
    # Download the data files
    utils.get_datafiles()

    # Check a slice of the image
    # img = skimage.io.imread("data/sandpack.tif")
    # plt.imshow(img[256], cmap="gray")
    # plt.colorbar()
    # plt.show()