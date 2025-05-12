import numpy as np
from scipy.optimize import curve_fit

class BeamFit:
    """
    Class to fit Gaussian and Hyperbolic functions to beam profile images.
    
    Attributes
    ----------
    image : np.ndarray
        2D array representing the beam profile image
    zpos : float
        Z-position of the image
    height : float
        Maximum intensity value in the beam profile
    xwaist : tuple
        Waist and error of the beam in the x-direction
    ywaist : tuple
        Waist and error of the beam in the y-direction

    Methods
    -------
    _find_index()
        Finds the coordinates of the maximum intensity value in the beam profile.
    
    _find_vector()
        Extracts the row and column vectors corresponding to the center of the beam profile.
    
    _extract_widths()
        Extracts the widths (standard deviations) of the beam in both x and y dimensions.
    
    _calculate_width(vector:np.ndarray)
        Calculates the width (standard deviation) of the beam along a given vector.
    
    _moments()
        Extracts the moments (height, center, and width) of the beam profile.
    
    _gaussian(x:list[int], height:float, centre:float, sigma:float) -> np.ndarray
        Defines the Gaussian function used for fitting the beam profile.
    
    fit() -> tuple
        Fits Gaussian functions to the beam profile in both x and y directions and returns the fit parameters and their errors.
    
    create_gaussian() -> tuple
        Generates the Gaussian profile based on the fitted parameters for both x and y directions.
    
    _dimensions() -> tuple
        Returns the x and y dimensions of the beam profile image.
    
    _hyperbolic(z:list[int], waist:float, z_0:float, wavelength:float, n:float) -> np.ndarray
        Defines the Hyperbolic function used to model the waist variation with z-position.
    
    fit_hyperbolic(beamfits:list, wavelength:float, n:float=1.003) -> tuple
        Fits Hyperbolic functions to the beam waist data for a list of beam profiles and returns the fit parameters and errors.
    
    _get_waist(dimension:str='x') -> tuple
        Extracts the waist (and its error) of the beam along the specified dimension (x or y).
    """
    
    def __init__(self,
                 image:np.ndarray,
                 zpos:float,
                 pix:tuple,
                 model:str='gaussian'
                 ) -> None:
        """
        Initializes the BeamFit class with a beam profile image and its z-position.
        
        Parameters
        ----------
        image : np.ndarray
            2D array representing the beam profile image.
        zpos : float
            Z-position of the image
        pix : tuple
            Pixel dimensions for chip used in beam image
        """    
        self.image = image
        self.zpos = zpos
        self.pix = pix
        self.model = model
        self.height = np.amax(self.image)
        self.floor = np.amin(self.image)
        self.xwaist = self._get_waist('x')
        self.ywaist = self._get_waist('y')

    def _find_index(self) -> int:
        """
        Finds the coordinates of the maximum intensity value in the beam profile.
        
        Returns
        -------
        tuple : (index_x, index_y)
            Indices of the center of the beam profile.
        """
        # if no image
        if self.image is None or self.image.sum() == 0:
            print("Image is empty - returning centre indexes")
            return self.image.shape[0] // 2, self.image.shape[1] // 2
    
        if self.model == 'gaussian':
            x_i, y_i = np.indices(self.image.shape)
            i_tot = np.sum(self.image)
            
            index_x = int(round(np.sum(x_i*self.image) / i_tot))
            index_y = int(round(np.sum(y_i*self.image) / i_tot))

            return index_x, index_y
        else:
            return np.unravel_index(np.argmax(self.image), self.image.shape)
    
    def _find_vector(self) -> tuple:
        """
        Extracts the row and column vectors corresponding to the center of the beam profile.
        
        Returns
        -------
        tuple : (row, col)
            Row and column vectors at the center of the beam profile.
        """
        index_x, index_y = self._find_index()
        row = self.image[:, index_y]
        col = self.image[index_x, :]
    
        return row, col
    
    def _calculate_width(self,
                         vector: np.ndarray
                         ) -> float:
        """
        Calculates the width (standard deviation) of the beam along a given vector,
        ignoring dark regions (zero-intensity pixels).
        
        Parameters
        ----------
        vector : np.ndarray
            The 1D vector (row or column) representing the beam profile.
        
        Returns
        -------
        float : The calculated beam width (standard deviation 
        or FWHM depending on model).
        """
        if self.model == 'gaussian':
            positions = np.arange(len(vector))  # Pixel indices
            nonzero_mask = vector > 0  # Mask to ignore dark pixels

            if not np.any(nonzero_mask):
                return 0  # If no beam is present, return 0 width

            vector_nonzero = vector[nonzero_mask]  # Keep only nonzero intensity values
            positions_nonzero = positions[nonzero_mask]  # Corresponding positions

            total_intensity = np.sum(vector_nonzero)
            mean_position = np.sum(positions_nonzero * vector_nonzero) / total_intensity

            variance = np.sum(vector_nonzero * (positions_nonzero - mean_position) ** 2) / total_intensity
            return np.sqrt(variance)  # Standard deviation
        else:
            half_max = np.max(vector) / 2
            indices = np.where(vector >= half_max)[0]
            # Profile too narrow or noisy
            if len(indices) < 2:
                return 0.0  
            return indices[-1] - indices[0]  # Width at half max

    def _extract_widths(self) -> tuple:
        """
        Extracts the widths (standard deviations) of the beam in both x and y dimensions.
        
        Returns
        -------
        tuple : (sigma_x, sigma_y)
            Widths (standard deviations) of the beam along the x and y dimensions
        """
        row, col = self._find_vector()
        return self._calculate_width(row) * self.pix[0], self._calculate_width(col) *  self.pix[1]

    def _moments(self) -> tuple:
        """
        Extracts the moments (height, center, and width) of the beam profile.
        
        Returns
        -------
        tuple : (height, index_x, sigma_x, index_y, sigma_y, floor)
            The height, center indices, widths (sigma) of the beam profile and height offset.
        """
        # handle case where there are multiple 'max' values
        index_x, index_y = self._find_index()
        # extract widths along the 2 dimensions
        sigma_x, sigma_y = self._extract_widths()

        return self.height, index_x * self.pix[0], sigma_x, index_y * self.pix[1], sigma_y, self.floor

    @staticmethod
    def _gaussian(x:list[int],
                  height:float,
                  centre:float,
                  sigma:float,
                  y_0:float
                  ) -> np.ndarray:
        """
        Defines the Gaussian function used for fitting the beam profile.
        
        Parameters
        ----------
        x : list[int]
            Array of x-values (positions) to evaluate the Gaussian function.
        height : float
            The peak value of the Gaussian (maximum intensity).
        centre : float
            The center position of the Gaussian (mean).
        sigma : float
            The standard deviation (width) of the Gaussian.
        
        Returns
        -------
        np.ndarray : Gaussian function evaluated at the given x values.
        """
        return height * np.exp(-(np.power(x - centre, 2) / (2 * sigma ** 2))) + y_0
    
    @staticmethod
    def _lorentzian(x:list[int],
                  height:float,
                  centre:float,
                  gamma:float,
                  y_0:float
                  ) -> np.ndarray:
        """
        Defines the Lorentzian function used for fitting the beam profile.
        
        Parameters
        ----------
        x : list[int]
            Array of x-values (positions) to evaluate the Lorentzian function.
        height : float
            The peak value of the Lorentzian (maximum intensity).
        centre : float
            The center position of the Lorentzian (mean).
        gamma : float
            The standard deviation (width) of the Lorentzian.
        
        Returns
        -------
        np.ndarray : Lorentzian function evaluated at the given x values.
        """
        return (height * ((0.5*gamma)**2/((x-centre)**2 + (0.5*gamma)**2))) + y_0
    
    def fit(self) -> tuple:
        """
        Fits Gaussian or Lorentzian profiles to the beam in both x and y directions.
        
        Returns
        -------
        tuple : (fit_data, fit_err)
            fit_data : Array of fit parameters (height, center, sigma) for x and y.
            fit_err : Array of errors for the Gaussian fit parameters for x and y.
        """
        model_funcs = {'gaussian': self._gaussian, 
                       'lorentzian': self._lorentzian}
        if self.model not in model_funcs:
            raise ValueError(f"Invalid type '{self.model}'. Please choose from {model_funcs.keys()}")
        
        params = self._moments()
        # extract data along index of maximum value
        row, col = self._find_vector()
        # generate positional arguments for gaussian
        x, y = self._dimensions()
        bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
        try:
            # fit gaussian to data and return probability
            fit_x, cov_x = curve_fit(model_funcs[self.model], x, 
                                        row, p0=(params[0], params[1], params[2], params[5]), bounds=bounds)
            fit_y, cov_y = curve_fit(model_funcs[self.model], y, 
                                        col, p0=(params[0], params[3], params[4], params[5]), bounds=bounds)
            x_err = np.sqrt(np.diag(cov_x))
            y_err = np.sqrt(np.diag(cov_y))
            # condense fit data into array for output
            return np.array([fit_x, fit_y]), np.array([x_err, y_err])
    
        except RuntimeError as e:
            print(f"Fit failed: {e}! Returning zeros.")
            return np.zeros((2, 3)), np.zeros((2, 3))
        
    def create_profile(self) -> tuple:
        """
        Generates the appropriate profile based on the fitted parameters for both x and y directions.
        
        Returns
        -------
        tuple : (x_profile, y_profile)
            x_profile : Profile along the x-axis.
            y_profile : Profile along the y-axis.
        """
        lengths = self._dimensions()
        if self.model == 'gaussian':
            x = self._gaussian(lengths[0], *self.fit()[0][0])
            y = self._gaussian(lengths[1], *self.fit()[0][1])
        else:
            x = self._lorentzian(lengths[0], *self.fit()[0][0])
            y = self._lorentzian(lengths[1], *self.fit()[0][1])
        
        return x, y
    
    def _dimensions(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the x and y coordinate arrays in real-world dimensions.

        Returns
        -------
        tuple : (x, y)
            x : Array of x-values in real-world units.
            y : Array of y-values in real-world units.
        """
        x_pixels, y_pixels = self.image.shape
        x = np.linspace(0, x_pixels * self.pix[0], x_pixels)
        y = np.linspace(0, y_pixels * self.pix[1], y_pixels)
        
        return x, y

    @staticmethod
    def _hyperbolic(z: np.ndarray,
                    waist:float,
                    z_0:float,
                    wavelength:float,
                    n:float
                    ) -> np.ndarray:
        """
        Defines the Hyperbolic function used to model the waist variation with z-position.
        
        Parameters
        ----------
        z : list[int]
            Array of z-positions.
        waist : float
            Waist size at the focal point.
        z_0 : float
            Reference z-position (e.g., the position of the beam waist).
        wavelength : float
            Wavelength of the beam.
        n : float
            Refractive index.
        
        Returns
        -------
        np.ndarray : Hyperbolic function evaluated at the given z-positions.
        """        
        z_R = BeamFit._z_R(waist, wavelength, n)
        return waist * np.sqrt(1 + ((z-z_0)/z_R)**2)
    
    @staticmethod
    def _z_R(waist,
             wavelength,
             n
             ) -> float:
        
        return (np.pi * n * waist**2) / wavelength

    @staticmethod
    def fit_hyperbolic(beamfits,
                       wavelength,
                       n=1.003,
                       ) -> tuple:
        """
        Fits Hyperbolic functions to the beam waist data for a list of beam profiles.
        
        Parameters
        ----------
        beamfits : list
            List of BeamFit objects containing the beam waist data.
        wavelength : float
            Wavelength of the beam.
        n : float, optional
            Refractive index, by default 1.003.
        
        Returns
        -------
        tuple : (fit_data, fit_err)
            fit_data : Fit parameters (waist size, z-position, refractive index).
            fit_err : Errors associated with the fit parameters.
        """
        z = [beam.zpos for beam in beamfits]
        w_x = [beam.xwaist[0] for beam in beamfits]
        w_y = [beam.ywaist[0] for beam in beamfits]
        w = [w_x, w_y]
        i = [np.argmin(w_x), np.argmin(w_y)]
    
        results = []
        for index, loc in enumerate(i):
            p0 = [w[index][loc], z[loc]]
            bounds = ([0, -np.inf], [1.2*w[index][loc], np.inf])
            try:
                 fit, covariance = curve_fit(
                     lambda z, waist, z_0: BeamFit._hyperbolic(z, waist, z_0, wavelength, n),
                     z, w[index], p0=p0, method='trf', bounds=bounds
                     )
                 fit_err = np.sqrt(np.diag(covariance))
                 results.append((fit, fit_err))
            except RuntimeError:
                print(f'Hyperbolic fit failed for {index}-axis. Returning Zeros.')
                results.append((np.zeros(2), np.zeros(2)))

        return results
        
    def _get_waist(self,
                   dimension:str='x'
                   ) -> tuple:
        """
        Extracts the waist (and its error) of the beam along the specified dimension (x or y).
        
        Parameters
        ----------
        dimension : str, optional
            Dimension to extract waist from ('x' or 'y'), by default 'x'.
        
        Returns
        -------
        tuple : (waist, error)
            Waist size and error for the specified dimension.
        """            
        fit_data, fit_err = self.fit()

        if dimension == 'x':
            return np.abs(self._to_waist(fit_data[0][2])), (self._to_waist(fit_err[0][2]))
        else:
            return np.abs(self._to_waist(fit_data[1][2])), (self._to_waist(fit_err[1][2]))

    def _to_waist(self,
                  value:float
                  ) -> float:
        """
        Convert beam width to beam waist (1/e^2)
        """
        if self.model == 'gaussian':
            # gaussian
            return 2*value
        else:
            # lorentzian
            return value