
# Fit functions for the Beam_Profile script

import numpy as np
from numpy import pi
from scipy.optimize import curve_fit

wavelength = 1550e-6

def gaussian(x, height, centre, sigma):
    """
    Generates Gaussian with given parameters
    
    Parameters
    ----------

    x : 1D array 
        Positional arguments for gaussian
    height : 1D array
        Maximum value of gaussian
    centre : Single value
        Centre of Gaussian peak
    sigma : Single value
        Width of Gaussian

    Returns
    -------

    1D array of height values for the positional arguments given in x
    """
    return height * np.exp(-(np.power(x - centre, 2) / (2 * sigma ** 2)))

def hyperbolic(z, waist, z_0): 
    """
    Generate hyperbolic function for given beam waist, z positions and wavelength

    Parameters
    ----------

    z : 1D array 
        Positional arguments for hyperbolic 
    waist : Single Value
        Minimum y-value for the function
    z_0 : Single Value
        Position of waist

    Returns
    -------

    1D array of intensity values for the given positional arguments in z
    """
    return waist * np.sqrt(1 + ((((z[:] - z_0) * wavelength)/(np.pi * waist ** 2)) ** 2))

def sincsquare(x, a, x_0, sigma): 
    """
    Generate sinc^2 function for given amplitude and x positions

    Parameters
    ----------

    a : Single Value
        Peak y-value for the function
    x : 1D array 
        Positional arguments for sinc 
    x_0 : Single Value
        Position of maximum value

    Returns
    -------

    1D array of values for the given positional arguments in x
    """
    return a * (np.sin((x[:]-x_0)*sigma) / ((x[:]-x_0)*sigma))**2

# fit a gaussian to data by calculating its 'moments' (mean, variance, width, height)
def moments(data):
    """
    Calculates parameters of a gaussian function by calculating
    its moments (height, index_x, width_x, index_y, width_y)

    Parameters
    ----------

    data : 2D array 
        2-Dimensional array of intensity data

    Returns
    -------

    Extracts the following along a single array within the matrix - the
    x and y values will be calculated along the same index as the maximum value of the matrix

    1D array with the following:

    height : Single value
        Calculates maximum value of the array and returns the index's to internal function
    index_x : Single value
        Calculates index of peak value along one-dimension of the matrix
    sigma_x : Single value
        Calculates the width (FWHM) along the first array
    index_y : Single value
        Calculates index of peak value along the other dimension of the matrix
    sigma_y : Single value
        Calculates the width (FWHM) along the second array
    """

    # find height and centre of gaussian
    height = np.amax(data)
    centre = np.where(data == height)
    dim = np.size(centre)
    # handle case where there are multiple 'max' values
    if dim > 2:
        centre = np.array(centre)
        index = np.sum(centre, 1)
        index_x = round(index[0] / len(centre[0]))
        index_y = round(index[1] / len(centre[1]))
    else:
        index_x = int(centre[0])
        index_y = int(centre[1])

    # extract widths along the 2 dimensions
    row = data[:, index_y]
    col = data[index_x, :]
    sigma_x = np.sqrt(((row - height) ** 2).sum() / len(row))
    sigma_y = np.sqrt(((col - height) ** 2).sum() / len(col))

    return height, index_x, sigma_x, index_y, sigma_y

def fitgauss(data):
    """
    Returns seperate x-y Gaussian parameters from fit to 2D gaussian data
    (height, centre_x, width_x, centre_y, width_y)

    Calls to moments(data) in order to extract relevant parameters of the 
    2D gaussian data before finding the fit to the data. See scipy.optimize.curve_fit
    for more on data fitting.

    Parameters
    ----------

    data : 2D array 
        2-Dimensional array of intensity data

    Returns
    -------

    fit_data : 1D Array
        Fitted variables: height, sigma, mean
    fit_err : 1D Array
        Uncertainty in fitted variables
    """
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

def fithyp(z, beam_d, params=None, meth=None, lims=(-np.inf, np.inf)):
    """
    Returns the beam waist of an array of beam diameters at positions
    along z for a fundamental mode gaussian beam

    Parameters
    ----------

    z : 1D Array
        Position data corresponding to diameter data in beam_d
    beam_d : 1D Array
        Array of beam diameters according to positions in z
    params : 1D Array
        Guess values for hyperbolic function; waist, z_0
    meth : Single string {'lm', 'tf', 'dogbox'}, optional
        Method to use for optimisation. See 
        scipy.optimize.curve_fit for details
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to 
        no bounds. 
        See scipy.optimize.curve_fit for details

    Returns
    -------

    fit_data : 1D Array
        Fitted variables: height, sigma, mean
    fit_err : 1D Array
        Uncertainty in fitted variables
    """
    fit, success = curve_fit(hyperbolic, z, beam_d, p0=params, method=meth, bounds=lims)
    fit_err = np.sqrt(np.diag(success))
    return fit, fit_err

def fitsincsquare(x, y, params=None, meth=None, lims=(-np.inf, np.inf)):
    """
    Returns a series of values that fit to a sinc(x) function

    Parameters
    ----------

    x : 1D Array
        Position data corresponding to amplitude in y
    y : 1D Array
        Array of amplitude values according to positions in x
    params : 1D Array
        Guess values for sinc function; x_0, amplitude
    meth : Single string {'lm', 'tf', 'dogbox'}, optional
        Method to use for optimisation. See 
        scipy.optimize.curve_fit for details
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to 
        no bounds. 
        See scipy.optimize.curve_fit for details

    Returns
    -------

    fit_data : 1D Array
        Fitted variables: height, sigma, mean
    fit_err : 1D Array
        Uncertainty in fitted variables
    """
    fit, success = curve_fit(sincsquare, x, y, p0=params, method=meth, bounds=lims)
    fit_err = np.sqrt(np.diag(success))
    return fit, fit_err