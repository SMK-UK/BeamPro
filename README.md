# Beam Profile and Waist Finder
@SMK-UK 02/2025

## v.2.0

## code for analysing beam images (basic level)

What is new? Updated this to operate as a class enabling faster analysis with less on the front end.

- Take images of beam shape (Gaussian) over known distances
- Save these images in a folder along with a .csv file of the distance for each image
- Run BeamProcessor (units in mm)
- Enables calculation of the beam waist from the images taken
    BeamProcessor.fit_beam()
- Individual images can be plotted (raw or processed)
    BeamProcessor.plotter.plot(index)
- Fits can be plotted individually also
    BeamProcessor.plotter.plot_fit(index)

