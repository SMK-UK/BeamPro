from BeamProcessor import BeamProcessor as bp
# directory t images and distance files
dir = r"Examples\Images"
# initialise the class
beam = bp(directory=dir, chipsize=[9.6, 7.68], wavelength=1550E-6)
# plot raw image
beam.plotter.plot(0)
# plot processed image, cropped around the beam
beam.plotter.plot(0, crop=True)
# plot processed image with fit
beam.plotter.plot_fit(0, True)
# plot the overall fit
beam.plotter.plot_beam()
# print out the fitted waist
beam.fit_beam()
# print out data from individual images
beam.processed[0].zpos
beam.processed[0].xwaist
beam.processed[0].ywaist