from PIL import Image
from natsort import natsorted
import numpy as np
import os
import pandas as pd

def dir_interogate(path: str, extensions: tuple[str] or list[str], 
                   exceptions: tuple[str] or list[str] =None, folders: tuple[str]=None):
    """
    Interogate directory and extract all folders and files with 
    the specified extensions

    Parameters
    ----------

    path : string - main folder / directory to interrogate
    exts : tuple / list - file extensions to check for in directory
    exceptions : tuple / list - file extensions / strings to exclude

    Returns
    -------

    folder_list : list of folder names
    file_list : list of file names

    """
    save_files = False
    folder_list = []
    file_list = []
    # holder removes parent folder from lists
    holder = 0
    # walk through directory and extract all relevant files
    for root, dirs, files in natsorted(os.walk(path)):
        if holder == 1:
            if folders == None:
                # populate folder list
                folder_list.append(root)
                save_files = True
            elif(root.endswith(folders)):
                # populate selected folder list
                folder_list.append(root)
                save_files = True
            temp = []
            if save_files == True:
                for file in natsorted(files):
                    # check for file extension
                    if(file.endswith(extensions)):
                        if exceptions == None:
                            temp.append(file)
                        elif any([x in file for x in exceptions]):
                            continue
                        else:
                            temp.append(file)
                file_list.append(temp)
                save_files = False
        else:
            holder = 1

    if len(file_list) == 1:
        file_list = [file_name for sublist in file_list for file_name in sublist]

    return folder_list, file_list

def image_read(path: str, mode: str =None, convert=0):

    image = Image.open(path)
    if convert == 1:
        image = image.convert(mode=mode, matrix =None, dither=None, palette=0, colors=256)

    return image

def open_excel(path: str, seperators: str=','):
    """
    Open a given excel / csv file and generate list

    Parameters
    ----------
    path : file path
    
    Returns
    -------
    excel_data : pandas data frame 
    
    """
    temp_df = pd.read_csv(path, sep=seperators)
    excel_data = [temp_df[x].values.tolist() for x in temp_df]

    if len(excel_data) == 1:
        excel_data = [value for sublist in excel_data for value in sublist]

    return excel_data