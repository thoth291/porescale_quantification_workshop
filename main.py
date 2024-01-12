import matplotlib.pyplot as plt
from src import utils
from src.analysis import ImageQuantifier
import pandas as pd

if __name__ == "__main__":
    # Download the data files
    # utils.get_datafiles()

    for image in ['beadpack', 'castlegate', 'mtgambier', 'sandpack']:
        # Check a slice of the image
        img = ImageQuantifier(f"data/{image}.tif")
        # img.plot_slice()
        img.run_analysis(heterogeneity_kwargs={'no_radii': 20, 'no_samples_per_radius': 500}, ev_kwargs={'cube_size': 256},
                         to_file_kwargs={'filetype': 'parquet'})

    minkowski = pd.read_parquet("data/image_characterization_results/minkowski.parquet")
    hetero = pd.read_parquet("data/image_characterization_results/heterogeneity.parquet")
    subset = pd.read_parquet("data/image_characterization_results/subsets.parquet")

    print(minkowski)
    print(hetero.head())
    print(subset.head())