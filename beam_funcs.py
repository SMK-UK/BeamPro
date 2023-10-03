from PIL import Image
from natsort import natsorted
import numpy as np
import os
import pandas as pd

def dir_interogate(path: str, extensions: tuple[str,...] = (), 
                   exceptions: tuple[str,...] = (), 
                   folders: tuple[str,...] = ()):
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
    for root, dirs, files in natsorted(os.walk(path)):

        if dirs:
            dirs = natsorted(dirs)
            if not folders:
                folder_list = dirs
            else:
                folder_list = [folder for folder in dirs 
                               if folder in folders]
            if exceptions:
                folder_list = [folder for folder in folder_list
                               if not any([x in folder for x in exceptions])]

        if not dirs:
            temp_files = []
            if not folders:
                temp_files = files
            elif any([x in os.path.split(root) for x in folders]):
                temp_files = files
            if exceptions:
                temp_files = [file for file in temp_files
                              if not any([x in file for x in exceptions])]
            if extensions:
                temp_files = [file for file in temp_files
                              if file.endswith(extensions)]
            if temp_files:
                file_list.append(natsorted(temp_files))

    if len(file_list) == 1:
        file_list = [file_name for sublist in file_list
                     for file_name in sublist]
    
    return folder_list, file_list

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