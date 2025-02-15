from BeamFit import BeamFit
import matplotlib.pyplot as mp
from numpy import arange, linspace, max, ndarray, transpose
from mpl_toolkits.axes_grid1 import make_axes_locatable

mp.style.use("signature.mplstyle")

class BeamPlot:
    """
    Class to plot the raw and processed beam profile images and the results of the beam waist fitting.

    Attributes
    ----------
    raw : list
        List of raw beam images.
    processed : list
        List of processed beam images.
    wavelength : float
        Wavelength of the laser.
    n : float, optional
        Refractive index of the medium (default is 1.003).

    Methods
    -------
    __init__(self, raw, processed, pixsize, wavelength, n=1.003):
        Initializes the BeamPlot instance with the necessary data.

    _dimensions(data):
        Returns the x and y dimensions of the image contained in data.
        
    _generate_z(extent, offset):
        Generates a range of z-positions for the beam waist fitting.
        
    plot(self, index, processed=False):
        Plots the image at the given index (either raw or processed).
        
    plot_fit(self, index):
        Plots the image at the given index with the Gaussian fit.
        
    plot_beam(self):
        Plots the beam waist fit results in both the x and y directions.
    """

    def __init__(self,
                 raw,
                 processed,
                 pixsize,
                 wavelength,
                 n=1.003
                 ) -> None:
        """
        Initializes the BeamPlot instance with the necessary data.

        Parameters
        ----------
        raw : list
            List of raw beam images.
        processed : list
            List of processed beam images.
        pixsize : list
            Pixel size for the images.
        wavelength : float
            Wavelength of the laser.
        n : float, optional
            Refractive index of the medium (default is 1.003).

        Raises
        ------
        ValueError
            If any of the required parameters are missing.
        """
        if not raw:
            raise ValueError("No raw images provided")
        if not processed:
            raise ValueError("No processed images provided")
        if not pixsize:
            raise ValueError("No pixel sizes provided")
        if not wavelength:
            raise ValueError("Wavelength not provided")
        
        self.raw = raw
        self.processed = processed
        self.wavelength = wavelength
        self.n = n

    @staticmethod
    def _dimensions(data) -> tuple:
        """
        Returns the x and y dimensions of the image at the given index.
        
        Parameters
        ----------
        data : 
            Index of the image for which to calculate the dimensions.
        
        Returns
        -------
        tuple : (x, y)
            x and y coordinates corresponding to the image dimensions.
        """
        lengths = data.image.shape
        x = arange(1, lengths[0]+1) * data.pix[0]
        y = arange(1, lengths[1]+1) * data.pix[1]

        return x, y
    
    @staticmethod
    def _generate_z(extent,
                    offset
                    ) -> ndarray:
        """
        Generates a range of z-positions for the beam waist fitting.

        Parameters
        ----------
        extent : float
            The maximum extent for the z-range.
        offset : float
            The offset to center the z-range.

        Returns
        -------
        ndarray
            Array of z-positions for plotting the fit.
        """
        return linspace(start=-extent, stop=extent, num=1000, endpoint=True) + offset
    
    @staticmethod
    def _crop(data):

        fit = data.fit()[0]
        centre_x = round(fit[0][1], 2)
        centre_y = round(fit[1][1], 2)
        x = round(data.xwaist[0], 2)
        y = round(data.ywaist[0], 2)
        limits = [centre_x - 2*x, centre_x + 2*x, centre_y - 2*y, centre_y + 2*y]

        return limits 

    def plot(self,
             index,
             processed=False,
             crop=True
             ):
        """
        Plots the image at the given index (either raw or processed).
        
        Parameters
        ----------
        index : int
            Index of the image to plot.
        processed : bool, optional
            Whether to plot the processed image (default is False).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        fig, ax = mp.subplots()

        if processed:
            data = self.processed[index]
            print(data)
            x, y = self._dimensions(data)
            ax.imshow(transpose(data.image), extent=[0, max(x), max(y), 0], cmap='viridis')
            if crop:
                lims = self._crop(data)
                ax.set_xlim(left=lims[0], right=lims[1])
                ax.set_ylim(top=lims[2], bottom=lims[3])
        else:
            image = self.raw[index]
            ax.imshow(image, cmap='viridis')
        
        ax.set(xlabel='Chip size (mm)', ylabel='Chip size (mm)')

        return fig, ax
    
    def plot_fit(self,
                 index,
                 crop=True
                 ):
        """
        Plots the image at the given index with the Gaussian fit.
        
        Parameters
        ----------
        index : int
            Index of the image to plot.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        # get relevant data from BeamFit Object
        data = self.processed[index]
        x, y = self._dimensions(data)
        data_x, data_y = data._find_vector()
        amp_x, amp_y = data.create_gaussian()
        # create figure and axes
        fig, ax = mp.subplots()
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", 1, pad=0.1, sharex=ax)
        ax_right = divider.append_axes("right", 1, pad=0.1, sharey=ax)
        # remove corresponding axes labels
        ax_top.xaxis.set_tick_params(labelbottom=False)
        ax_right.yaxis.set_tick_params(labelleft=False)
        # plot the x fit and real data
        ax_top.plot(x, data_x, label=' X Data', alpha=0.75)
        ax_top.plot(x, amp_x, linestyle ='--', label='Fit in X')
        # plot the y fit and real data
        ax_right.plot(data_y, y, label='Y Data', alpha=0.75)
        ax_right.plot(amp_y, y, linestyle ='--', label='Fit in Y')
        # crop if required
        if crop:
            lims = self._crop(data)
            ax.set_xlim(left=lims[0], right=lims[1])
            ax.set_ylim(top=lims[2], bottom=lims[3])
        # format labels
        ax.set(xlabel='Chip size (mm)', ylabel='Chip size (mm)')
        ax_top.set(ylabel='Intensity (AU)')
        ax_right.set(xlabel='Intensity (AU)')
        # legend
        ax_top.legend(loc='best', bbox_to_anchor=(1,1))

        ax.imshow(transpose(data.image), extent=[0, max(x), max(y), 0], cmap='viridis')

        return fig, ax
    
    def plot_beam(self):
        """
        Plots the beam waist fit results in both the x and y directions.
        
        This function uses the hyperbolic fit method from BeamFit to determine
        the beam waist at different z-positions and plots the results, including
        error bars and the fit lines.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        # Extract beam waist data
        z = [beam.zpos for beam in self.processed]
        w_x = [beam.xwaist[0] for beam in self.processed]
        w_y = [beam.ywaist[0] for beam in self.processed]
        x_err = [beam.xwaist[1] for beam in self.processed]
        y_err = [beam.ywaist[1] for beam in self.processed]
        # Perform hyperbolic fit to the waist data
        x_results, y_results = BeamFit.fit_hyperbolic(self.processed, self.wavelength, self.n)
        print('xwaist', x_results[0][0], x_results[0][1])
        print('ywaist', y_results[0][0], y_results[0][1])
        # Calculate Rayleigh range for both x and y directions
        zR_x = BeamFit._z_R(x_results[0][0], self.wavelength, self.n)
        zR_y = BeamFit._z_R(y_results[0][0], self.wavelength, self.n)
        print('Rayleigh', zR_x)
        print('Rayleigh', zR_y)
        # Generate z-position values for the plot (extended Rayleigh range)
        z_x = self._generate_z(1.2*zR_x, x_results[0][1])
        z_y = self._generate_z(1.2*zR_y, y_results[0][1])
        # Plot the beam waist data and fits
        fig, ax = mp.subplots()
        ax.errorbar(z, w_x, yerr=x_err, color='C0', label='$\omega_{0}$ x', fmt='.')
        ax.errorbar(z, w_y, yerr=y_err, color='C1', label='$\omega_{0}$ y', fmt='.')
        ax.plot(z_x, BeamFit._hyperbolic(z_x, x_results[0][0], 
                                         x_results[0][1], self.wavelength, self.n), 
                                         label='$\omega_{0}$ x-fit', linestyle='--')
        ax.plot(z_y, BeamFit._hyperbolic(z_y, y_results[0][0], 
                                         y_results[0][1], self.wavelength, self.n), 
                                         label='$\omega_{0}$ y-fit', linestyle='--')
        # Set labels and legend
        ax.set(xlabel=('Z Position (mm)'), ylabel=('1/e$^{2}$ Beam Waist (mm)'))
        ax.legend(loc='best')

        return fig, ax