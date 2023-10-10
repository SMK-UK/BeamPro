from PIL import Image
from natsort import natsorted
import numpy as np
import os
import pandas as pd
from fit_funcs import fitgauss

def dir_interogate(path: str, extensions: tuple[str,...] = (), 
                   exceptions: tuple[str,...] = (), 
                   choose_folders: tuple[str,...] = ()):
    """
    Interogate directory and extract all folders and files with 
    the specified extensions

    Parameters
    ----------

    path : string - main folder / directory to interrogate
    exts : tuple / list - file extensions to check for in directory
    exceptions : tuple / list - file extensions / strings to exclude
    folders : list - selected folders to extract from

    Returns
    -------

    folder_list : list of folder names
    file_list : list of file names

    """
    folder_list = []
    file_list = []
    for root, folders, files in natsorted(os.walk(path)):

        temp_files = []

        if folders:
            folders = natsorted(folders)
            if choose_folders:
                folder_list = [folder for folder in folders 
                                if folder in choose_folders]
            else:
                folder_list = folders
        else:
            if not choose_folders:
                temp_files = files
            elif any ([x in os.path.split(root) for x 
                        in choose_folders]):
                    temp_files = files
            if exceptions:
                temp_files = [file for file in temp_files 
                              if not any([x in file for x in exceptions])]
            if extensions:
                temp_files = [file for file in temp_files 
                                if file.endswith(extensions)]
            if temp_files:
                file_list.append(temp_files)

    if len(file_list) == 1:
        file_list = [file_name for sublist in file_list
                        for file_name in sublist]
    
    return folder_list, file_list

def get_pix_size(image_file: str, chip_dims: []):
    """
    Open a given excel / csv file and generate list

    Parameters
    ----------
    image_file : path for image file to check dimensions
    chip_dims : x,y dimensions (mm) of chip in camera
    
    Returns
    -------
    pixel dimensions : number of pixels per mm [x, y] 
    """
    base_image = Image.open(image_file)
    im_size = base_image.size

    return [chip_dims[0]/ im_size[0], chip_dims[1]/ im_size[1]]

def norm_image(image_file: str, bkd_file: str):
    """
    Normalise an image by subtracting background noise

    Parameters
    ----------
    image_file : path for image file to check dimensions
    bkd_file : path for background image to subtract from image_file
    
    Returns
    -------
    normalised : normalised image as array
    """

    normalised = np.empty([])
    if not image_read(image_file).mode in ('I;16', 'L'):
        beam_image = np.float64(np.transpose(np.asarray(image_read(path=image_file, mode='L', convert=1))))
        beam_bkd = np.float64(np.transpose(np.asarray(image_read(path=bkd_file, mode='L', convert=1))))
    else:
        beam_image = np.float64(np.transpose(np.asarray(image_read(path=image_file, convert=0))))
        beam_bkd = np.float64(np.transpose(np.asarray(image_read(path=bkd_file, convert=0))))
    
    normalised = np.absolute(beam_image - beam_bkd)
    normalised *= 255/normalised.max()

    return normalised

def image_read(path: str, mode: str =None, convert=0):
    """
    Open a given excel / csv file and generate list

    Parameters
    ----------
    path : file path
    mode : image type
    convert : Bool to convert image to type 'mode' 
    
    Returns
    -------
    image : PIL object of the image
    """
    image = Image.open(path)
    if convert == 1:
        image = image.convert(mode=mode, matrix =None, dither=None,
                              palette=0, colors=256)

    return image

def open_excel(path: str, seperators: str=','):
    """
    Open a given excel / csv file and generate list

    Parameters
    ----------
    path : file path
    
    Returns
    -------
    excel_data : list of the extracted data 
    """
    temp_df = pd.read_csv(path, sep=seperators)
    excel_data = [temp_df[x].values.tolist() for x in temp_df]

    if len(excel_data) == 1:
        excel_data = [value for sublist in excel_data for value in
                      sublist]

    return excel_data