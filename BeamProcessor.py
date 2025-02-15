from PIL import Image
from natsort import natsorted
import numpy as np
import pandas as pd
import os, sys

from BeamFit import BeamFit
from BeamPlot import BeamPlot

class BeamProcessor:
    """
    Class to process beam profile images and extract data
    
    Attributes
    ----------
    chipsize : list[float]
        Size of chip in mm
    directory : str
        Directory containing images
    wavelength : float
        Wavelength of beam in mm
    images : list[Image.Image]
        List of images loaded from directory
    processed : list[np.ndarray]
        List of processed images
    pix : list[float]
        Pixel size of images in mm
    zpos : list
        Z-position data
    
    Methods
    -------
    _get_zpos()
        Extracts z-position data from csv file

    _get_fnames(extensions:str='csv')
        Extracts filenames from directory with given extensions

    _join(fname:str)
        Joins filename to directory path

    _load_images()
        Loads images from directory

    _process_images()
        Processes images by normalising them

    _get_pix_size()
        Calculates pixel size of images in mm

    _image_read(path:str, mode:str=None, convert=0)
        Reads image from path and converts to given mode

    _img_to_numpy(fname)
        Converts image to numpy array

    _norm_image(image_file:str, bkd_file:str)
        Normalises image by subtracting background

    _normalised(image:np.ndarray, background:np.ndarray)
        Normalises image by subtracting background

    _open_excel(path:str, seperators:str=',')
        Opens excel file and extracts data
    
    """
    def __init__(self,
                directory:str,
                chipsize:list[float],
                wavelength:float=1550E-6,
                n:float=1.003
                ) -> None:
        
        self.chipsize = chipsize
        self.directory = directory
        self.wavelength = wavelength
        self.n = n
        self.zpos = self._get_zpos()
        self.images = self._load_images()
        self.pix = self._get_pix_size()
        self.processed = self._process_images()
        self.plotter = BeamPlot(self.images, self.processed, self.pix, self.wavelength, self.n)

    def _join(self, 
              fname:str
              )->str:
        """
        Joins filename to directory path
        
        Parameters
        ----------
        fname : str
            Filename to join to directory path
            
        Returns
        -------
        str : Full path to filename

        """
        return os.path.join(self.directory, fname)  

    def _get_fnames(self, 
                    extensions:str=['xlsx', 'xls', 'csv']
                    )->list[str]:
        """
        Extracts filenames from directory with given extensions
        
        Parameters
        ---------- 
        extensions : str, optional
            File extensions to search for in directory, by default 
            ['xlsx', 'xls', 'csv']
            
        Returns
        -------
        list : List of filenames with given extensions

        """
        files = natsorted(os.listdir(self.directory))
        fnames = [x for x in files if any([ext in x for ext in extensions])]

        return [self._join(name) for name in fnames]

    def _open_excel(self,
                    path: str,
                    separator: str = ','
                    )->np.ndarray:
        """
        Opens excel file and extracts data

        Parameters
        ----------
        path : str
            Path to excel file
        separator : str, optional
            Separator for data, by default ','
        
        Returns
        -------
        np.ndarray : Numpy array of data extracted from excel file

        """
        ext = os.path.splitext(path)[-1].lower()

        if ext == ".csv":
            df = pd.read_csv(path, sep=separator, header=0)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path, header=0, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return df.values.flatten()

    def _get_zpos(self
                  )->list:
        """
        Extracts z-position data from csv file

        Returns
        -------
        list : z-position data
        
        """
        paths = self._get_fnames()
        if not paths:
            sys.exit("No csv file found for z-position data")

        return self._open_excel(paths[0])
    
    def _load_images(self
                     )->list[Image.Image]:
        """
        Loads images from directory
        
        Returns
        -------
        list : List of images loaded from directory
        
        """
        paths = self._get_fnames(['tif', 'bmp'])

        return [Image.open(path) for path in paths]
    
    def _get_pix_size(self
                      )->list[float]:
        """
        Calculates pixel size of images in mm
        
        Returns
        -------
        list : List of pixel sizes for x and y dimensions
        
        """
        im_size = self.images[0].size

        return [self.chipsize[0]/im_size[0], self.chipsize[1]/im_size[1]]
    
    def _image_convert(self,
                       image:Image.Image,
                       mode:str='L'
                       )->Image.Image:
        """
        Converts image to given mode
        
        Parameters
        ----------
        image : Image.Image
            Image object to convert
        mode : str
            Mode to convert image to
            
        Returns
        -------
        Image.Image : Converted image object
        
        """
        return image.convert(mode=mode, matrix =None, dither=None,
                             palette=0, colors=256)

    def _img_to_numpy(self,
                      image
                      )->np.ndarray:
        """
        Converts image to numpy array
        
        Parameters
        ----------
        fname : str
            Path to image file
        
        Returns
        -------
        np.ndarray : Numpy array of image data
        
        """
        if image.mode in ('I;16', 'L'):
           return np.float64(np.transpose(np.asarray(image)))
        else:    
            return np.float64(np.transpose(np.asarray(self._image_convert(image))))
            
    def _norm_image(self,
                    image_file:str,
                    bkd_file:str
                    )->np.ndarray:
        """
        Normalises image by subtracting background
        
        Parameters
        ----------
        image_file : str
            Path to image file
        bkd_file : str
            Path to background image file
        
        Returns
        -------
        np.ndarray : Normalised image as numpy array
        
        """
        beam_image = self._img_to_numpy(image_file)
        beam_bkd = self._img_to_numpy(bkd_file)

        return self._normalised(beam_image, beam_bkd)
    
    @staticmethod
    def _normalised(image:np.ndarray,
                    background:np.ndarray
                    )->np.ndarray:
        """
        Normalises image by subtracting background
        
        Parameters
        ----------
        image : np.ndarray
            Image data
        background : np.ndarray
            Background data
        
        Returns
        -------
        np.ndarray : Normalised image data
        
        """
        normalised = np.absolute(image-background)
        normalised *= 255/normalised.max()

        return normalised
    
    def _process_images(self
                        )->list[np.ndarray]:
        """
        Processes images by normalising them

        Returns
        -------
        list : List of processed images

        """
        processed = []
        num_images = len(self.images)
        if num_images % 2 != 0:
            print("Warning: Odd number of images detected. The last image will be ignored.")
            num_images -= 1  
            
        for m in range(num_images//2):
            processed.append(self._norm_image(self.images[2 * m], self.images[2 * m + 1]))
        
        return [BeamFit(image, z, self.pix) for image, z in zip(processed, self.zpos)]
    
    def fit_beam(self, verbose=True):
        """
        Fits the hyperbolic function to the beam waist data at different z-positions.
        
        Returns
        -------
        tuple : (fit_data, fit_err)
            fit_data : Fit parameters (waist size, z-position).
            fit_err : Errors associated with the fit parameters.
        """
        results = BeamFit.fit_hyperbolic(self.processed, self.wavelength, self.n)

        if verbose:
            print(f'x waist = {results[0][0][0]:.2f} pm {results[0][1][0]:.2f}mm \
                  located at {results[0][0][1]:.2f} pm {results[0][1][1]:.2f}mm')
            print(f'y waist = {results[1][0][0]:.2f} pm {results[1][1][0]:.2f}mm \
                  located at {results[1][0][1]:.2f} pm {results[1][1][1]:.2f}mm')

        return results