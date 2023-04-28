'''
Specific functions for handling spectroscopy data and analysing
'''
from math import log
import matplotlib.pyplot as mp
from natsort import natsorted
import numpy as np
import os, re
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fft, fftfreq
from scipy.signal import find_peaks

def check_len(data_lists: list[list]):
    """
    Checks nested list for each list lengths and populate a list of lists
    containing the corresponding length
    
    Parameters
    ----------

    lists : list of lists

    Returns
    -------

    boolean : True or False value depending on if the 
    lists are all the same length lengths : list or 
    single value of lengths for each list in lists
    """
    # creat list of all nested list lengths
    data_set_lengths = [len(data_set_child) for data_set in data_lists 
                        for data_set_child in data_set]
    # check set for different lengths and populate a list
    lengths = [x for x in set(data_set_lengths)]
    # if only one length all lengths are the same so boolean is True
    if len(lengths) == 1:
        boolean = True
        indexes = None
    else:
        boolean = False
        indexes = [[len(x) for x in data_set_child] for data_set_child
                    in data_lists]

    return boolean, indexes

def check_str(input_string: str):
    """
    Checks string for certain characters and flag if true
    
    Parameters
    ----------

    input_string : string 

    Returns
    -------

    logical : Boolean
    """
    # check input string for digits and flag True
    if any(char.isdigit() for char in input_string) == True:
        # search input_string for any of the following characters 
        # and spaces / indents
        char_allow = set("0123456789\n\t\r eE-+,.;")
        validation = set((input_string))
        # check if any allowed characters in the input string
        logical = validation.issubset(char_allow)
    else:
        logical = False

    return logical

def data_extract(paths: list[str], keys: list[str]=None, tail: int=1, 
                include: bool=True):
    """
    search a given path or list of paths for strings and extract the data from
    selected files depending on the discriminator (keys)

    Parameters
    ----------

    paths : file paths / path
    keys : list of key values to search for in path if required
    tail : int value 1 or 0 to search head or tail of path
    function : True or False to include the data with key or exclude
    
    Returns
    -------

    extracted_data : list of data read from path
    extracted_metadata : list of metadata read from path
    
    """
    extracted_data = []
    extracted_metadata = []
    if keys != None:
        for key in keys:
            extracted_data_children = []
            extracted_metadata_children = []
            for path in paths:
                if include == True:
                    # extract data from path if it contains the key
                    if key in os.path.split(path)[tail]:
                        extracted_data_children.append(open_data(path)[0])
                        extracted_metadata_children.append(open_data(path)[1])
                    else:
                        continue
                else:
                    if key in os.path.split(path)[tail]:
                        # extract data from path if it doesn't contain the key
                        extracted_data_children.append(open_data(path)[0])
                        extracted_metadata_children.append(open_data(path)[1])
                    else:
                        continue
            extracted_data.append(extracted_data_children)
            extracted_metadata.append(extracted_metadata_children)
    else:
        for path in paths:
            extracted_data.append(open_data(path)[0])
            extracted_metadata.append(open_data(path)[1])

    return extracted_data, extracted_metadata

def data_fft(times: list[list[int]], amplitudes: list[list[int]]):
    """
    Perform FFT calculation for amplitude component and generate the frequency
    from times.

    Parameters
    ----------

    times : 2D data array / list to use as reference
    amplitudes : 2D data array / list of transmission data

    Returns
    -------

    frequencies : list of frequency values from input time data
    ffts : list of fft calculated from input amplitude data
    """
    frequencies = []
    ffts = []
    for index in range(len(times)):
        temp_freq = []
        temp_fft = []
        # perform correction on the reference data and then calculate the OD
        for time_child in times[index]:
            N = len(time_child)
            T = time_child[1] - time_child[0]
            temp_fft.append(np.sqrt(np.abs(fft(amplitudes[index]))))
            temp_freq.append(fftfreq(N, T))
        ffts.append(temp_fft)
        frequencies.append(temp_freq)

    return frequencies, ffts

def data_shift(data_sets: list[list[list[int]]], shift: int):
    """
    Perform a shift for each value in a list of data

    Parameters
    ----------

    data_sets : 2D data array / list to perform shift on
    shift : value to shift data by

    Returns
    -------

    shifted_sets : list of shifted data 
    """
    shifted_sets = [[[value + shift for value in data_set_child]
                    for data_set_child in data_sets[index]] for 
                    index in range(len(data_sets))]

    return shifted_sets

def df_average(data_frames: list):
    """
    Average a set of data frames and return the averaged
    data frame
    
    Parameters
    ----------

    data_frames : list of data frames to average  

    Returns
    -------

    averaged_data : data frame containing the averaged data from
    data_frames
    """
    summed = 0
    for data_frames_child in data_frames:
        summed += data_frames_child
    averaged_data = summed / len(data_frames)

    return averaged_data

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

def excel_extract(folder_names: list[str], file_names: list[list[str]], average: bool=False):

    extracted_excel = [[open_excel(os.path.join(folder, file)) for file in file_names[index]] for index, folder in enumerate(folder_names)]

    return extracted_excel

def find_numbers(paths: list[str], tail: int=1):
    """
    Checks string for numbers
    
    Parameters
    ----------

    paths : path or list of paths

    Returns
    -------

    numbers : int or list of ints
    """
    # will return combined values i.e. 187 will be returned '187' and not
    #'1', '8', '7'
    numbers = [re.findall('\d+', os.path.split(path)[tail])[0]
               for path in paths]
    
    return numbers

def OD_calc(ref_data: list[list[int]], trans_data: list[list[int]],
            correction: bool=True, c_factor: int=1):
    """
    Perform OD calculation for transmission data and adjust the reference
    using the correction factor if neccesary

    Parameters
    ----------

    reference : 2D data array / list to use as reference
    transmission : 2D data array / list of transmission data
    correction : correction factor for the reference data

    Returns
    -------

    x : list of x values from input x data
    y : list of OD calculated from input y data
    """
    x = []
    y = []
    for index in range(len(ref_data)):
        temp_x = []
        temp_y = []
        # perform correction on the reference data and then calculate the OD
        for references in ref_data[index]:
            if correction == True:
                reference = [x*c_factor for x in references[1]]
            for transmission in trans_data[index]:
                temp_x.append(transmission[0])
                temp_y.append([log(a / b) for a, b in zip(reference, transmission[1])])
            x.append(temp_x)
            y.append(temp_y)

    return x, y

def open_data(path: str):
    """
    Open a given file and read the first two columns to a list
    Parameters
    ----------
    path : file path
    
    Returns
    -------
    data_list : list of data read from path
    metadata_list : list of metadata read from path
    
    """
    data_list = 0
    metadata_list = 0
    with open(path, 'r', newline='') as raw_file:
    # cycle through each row in the file
        for row in raw_file:
            # check for numerical data
            if check_str(row) == True:
                # generate list to populate with column data
                data_temp = [i for i in re.split(r"[\t|,|;]", row) if i != '']
                if data_list == 0:
                    data_list = [[] for _ in range(len(data_temp))]
                for index, data in enumerate(data_temp):
                    data_list[index].append((float(data)))
            else:
                # extract metadata
                metadata_temp = [i for i in re.split(r"[\t|,|;]", row) if i != '']
                if metadata_list == 0:
                    # generate list to populate with column metadata
                    metadata_list = [[] for _ in range(len(metadata_temp))]
                for index, metadata in enumerate(metadata_temp):
                    metadata_list[index].append((metadata))
        raw_file.close()

    return data_list, metadata_list

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

def peak_find(data_sets, tolerance=None, lims=None):
    """
    Find peaks in data

    Parameters
    ----------

    data_sets : 2D data array / list to perform shift on
    args : peak finding conditions (see scipy.signal.find_peaks)

    Returns
    -------

    peaks_data : indexes of peaks for each data set
    """  
    peaks_data = []

    for index in range(len(data_sets)):
        temp = []
        for data_sets_child in data_sets[index]:
            if lims == None:
                lower = 0
                upper = -1
            else:
                lower, upper = zoom(data_sets_child, lims)
            max = np.amax(data_sets_child[lower:upper])
            min = np.amin(data_sets_child[lower:upper])
            prom = 0
            if tolerance != None:
                prom = max * tolerance
            peak, _ = find_peaks(data_sets_child, prominence=prom)
            temp.append(peak)
        peaks_data.append(temp)

    return peaks_data

def plotter(x_data, y_data, data_indexes=None, keys=None, shifter=0,
            axis_lbls=None, sec_axis=True, save=False, 
            data_labels=None, lims=None, woi=None, res=80):
    """
    zoom in on a particular area of interest in a dataset

    Parameters
    ----------

    data : list / array - data to perform zoom
    bounds : tuple - lower and upper bounds of the region of interest

    Returns
    -------

    start, stop : start and stop index for the zoomed data

    """
    data_lbl = None
    if keys == None:
        keys = ['']

    for m, key in enumerate(keys):
        fig, ax = mp.subplots(figsize=(8, 5))
        ax.grid(True, color='silver', linewidth=0.5)
        if axis_lbls != None:
            ax.set_title('Halfwave Plate: ' + key)
            ax.set(xlabel=axis_lbls[0], ylabel=axis_lbls[1])
        if sec_axis == True:
            sec_ax = ax.secondary_xaxis('top', functions=
                                        (lambda x: 1e7/x, lambda x: 1e7/x))
            sec_ax.set_xlabel('Wavenumber (cm$^{-1}$)')
        if woi != None:
            for  vline in woi:
                ax.axvline(x=vline, linestyle='-.')

        shift = 0
        for n, x in enumerate(x_data[m]):
            x = np.asarray(x)
            y = np.asarray([value + shift for value in y_data[m][n]])
            colour = mp.cm.viridis(np.linspace(0, 1, len(x_data[m])))
            if lims == None:
                lower = 0
                upper = -1
            else:
                lower, upper = zoom(x, lims)
            if data_labels != None:
                data_lbl = os.path.split(data_labels[n])[1]
            ax.plot(x[lower:upper], y[lower:upper], color=colour[n], 
                    linestyle='-', linewidth=0.8, alpha=1, label=data_lbl)
            if data_indexes != None:
                data_index = [i for i in data_indexes[m][n]
                              if i >= lower and i <= upper]
                ax.plot(x[data_index], y[data_index], marker='x',
                        color=colour[n], linestyle='None', alpha=1,
                        label='_nolegend_')
            shift += shifter

        ax.legend(loc='best', fontsize=8)   

        if save == True:
            folder = os.path.split(data_labels[0])[0]
            region = str(round(x[lower])) + '_' + str(round(x[upper])) 
            + '_' + key
            name = folder + '\\' + region + '.png'

            fig.savefig(fname=name, dpi=res, format='png', bbox_inches='tight')
    
def read_files(folder_list, file_list):
    """
    Create file path in a folder for parsing data in the file

    Parameters
    ----------

    folder_list : list of folder names
    file_list : list of file names

    Returns
    -------

    path : file path name
    
    """
    path = []
    for index, folder in enumerate(folder_list):
        for file in file_list[index]:
            path.append(os.path.join(folder, file))
    
    return path

def search_paths(paths, keys):
    """
    Search paths for keys and then extract the file pathnames

    Parameters
    ----------

    paths : paths to search
    keys : key strings to look for in path name

    Returns
    -------

    key_paths : list of requested path names

    """
    key_paths = []
    excluded_paths = []
    for path in paths:
        if any([x in path for x in keys]):
            key_paths.append(path)
        else:
            excluded_paths.append(path)

    return key_paths, excluded_paths
    
def smooth_data(data_sets, sigma):
    """
    Perform smoothing of the data using gaussian filter 
    TODO include other smoothing operations

    Parameters
    ----------

    data_sets : 2D data array / list to perform shift on
    sigma : smoothing parameter

    Returns
    -------

    smoothed_sets : list of smoothed data
    """
    smoothed_sets = []
    for index in range(len(data_sets)):
        temp = []
        for data_set in data_sets[index]:
            temp.append(gaussian_filter(data_set, sigma))
        smoothed_sets.append(temp)

    return smoothed_sets

def split_lists(data_sets):
    """
    Split lists of lists of lists into constituent parts: ie. a 4 x 16 x 2 list will be returned as two 4 x 16 x 1 lists
    data_sets dimensionality: [Polarisations:[Temperature:[x,y]]]

    Parameters
    ----------

    data_sets : 3D data array / list to perform spit

    Returns
    -------

    data_lists : separated lists
    """

    data_lists = [[[[temperatures[0]] for temperatures in polarizations] for polarizations in data_sets]]
    for i in range(len(data_sets[0][0])-1):
        data_lists.append([[[temperatures[i+1]] for temperatures in polarizations] for polarizations in data_sets])

    return data_lists

def zoom(data, bounds):
    """
    zoom in on a particular area of interest in a dataset

    Parameters
    ----------

    data : list / array - data to perform zoom
    bounds : tuple - lower and upper bounds of the region of interest

    Returns
    -------

    start, stop : start and stop index for the zoomed data

    """
    for index, value in enumerate(data):
        if value <= bounds[0]:
            start = index
        if value <= bounds[1]:
            stop = index

    return start, stop
