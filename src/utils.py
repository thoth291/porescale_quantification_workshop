import subprocess
import numpy as np
import tifffile
import csv
import os

def runcmd(cmd, verbose = True, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

def download_file(url, filename):
    """
    Run a wget command to download a file from an HTTP or FTP server
    :param url: URL of the file to download
    :param filename: Downloaded file name
    :return: None
    """
    runcmd(f'wget -O {filename}.ubc {url}')

def get_datafiles(datapath='data'):
    """
    Download the data files contained in data/data_links.csv
    :return: None
    """
    with open(f'{datapath}/data_links.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        # Skip the header row
        next(csv_reader, None)
        for row in csv_reader:
            # Only download the data files if it does not already exist
            if not os.path.exists(f'{datapath}/{row[0]}.ubc'):
                # Download the file from Digital Rocks Portal
                download_file(row[1], f'{datapath}/{row[0]}')

            # Convert to tiff and invert pore and solid labels
            img = np.fromfile(f"{datapath}/{row[0]}.ubc", dtype=np.uint8).reshape((512, 512, 512))
            img = -1 * img + 1
            tifffile.imwrite(f"{datapath}/{row[0]}.tif", img.astype(np.uint8))

            # Remove the ubc file. Only keep the tiff file
            runcmd(f"rm -f {datapath}/{row[0]}.ubc")


def read_tiff(datapath: str) -> np.ndarray:
    """
    Reads a 3D numpy array from a tif file at the given filepath.
    :param datapath: The filepath of the tif file to read.
    :returns: np.ndarray: A 3D numpy array containing the data from the tif file.
    :raises: ValueError: If the given filepath does not point to a valid tif file.
    """

    if not os.path.exists(datapath):
        raise ValueError(f"File {datapath} does not exist.")

    extension = os.path.splitext(datapath)[1]
    if not (extension == ".tif" or extension == ".tiff"):
        raise ValueError(f"File {datapath} is not a tif file.")

    return tifffile.imread(datapath)

def write_csv(name, results):
    """
    Write a csv file containing the results of the analysis
    :name: Name of csv file
    :results: Results of the analysis to write to file
    :return: None
    """


