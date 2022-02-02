
# Fit functions for the Beam_Profile script

import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, height, centre, sigma):
    '''Generates Gaussian with given parameters'''
    return height * np.exp(-(np.power(x - centre, 2) / (2 * sigma ** 2)))

def hyperbolic(z, WAIST, WAVE, Z_0):
    '''generate hyperbolic function for given beam waist, z positions and wavelength'''  
    # laser wavelength
    #WAVE = 995e-6
    return WAIST * np.sqrt(1 + ((((z - Z_0) * WAVE)/ (np.pi * WAIST ** 2)) ** 2))

# fit a gaussian to data by calculating its 'moments' (mean, variance, width, height)
def moments(data):
    '''Calculates parameters of a gaussian function by calculating
    its moments (height, mean_x, width_x, mean_y, width_y'''
    height = np.amax(data)
    centre = np.where(data == height)
    dim = np.size(centre)
    # handle case where there are multiple 'max' values
    if dim > 2:
        centre = np.array(centre)
        mean = np.sum(centre, 1)
        mean_x = round(mean[0] / len(centre[0]))
        mean_y = round(mean[1] / len(centre[1]))
    else:
        mean_x = int(centre[0])
        mean_y = int(centre[1])

    row = data[:, mean_y]
    col = data[mean_x, :]
    sigma_x = np.sqrt(((row - height) ** 2).sum() / len(row))
    sigma_y = np.sqrt(((col - height) ** 2).sum() / len(col))
    return height, mean_x, sigma_x, mean_y, sigma_y

def fitgauss(data):
    '''Returns seperate x-y Gaussian parameters from fit to 2D gaussian data
     (height, mean_x, width_x, mean_y, width_y)'''
    params = moments(data)
    # extract data along index of maximum value
    x = data[:, params[3]]
    y = data[params[1], :]
    # fit gaussian to data and return probability
    fit_x, success_x = curve_fit(gaussian, np.arange(1, len(x) +1 ), x, p0=params[0:3])
    fit_y, success_y = curve_fit(gaussian, np.arange(1, len(y) +1 ), y, p0=(params[0], params[3], params[4]))
    x_err = np.sqrt(np.diag(success_x))
    y_err = np.sqrt(np.diag(success_y))
    # condense fit data into array for output
    fit_data = np.array([fit_x, fit_y])
    fit_err = np.array([x_err, y_err])
    return fit_data, fit_err

def fithyp(z, beam_d, WAVE, params):
    ''' Returns the beam waist of an array of beam diameters at positions
    along z for a fundamental mode gaussian beam '''
    fit, success = curve_fit(lambda z, beam_d: hyperbolic(z, beam_d, WAVE, Z_0), z, beam_d, p0=params, method='lm')
    fit_err = np.sqrt(np.diag(success))
    return fit, fit_err