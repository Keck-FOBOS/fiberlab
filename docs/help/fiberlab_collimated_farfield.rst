.. code-block:: console

    $ fiberlab_collimated_farfield -h
    usage: fiberlab_collimated_farfield [-h] [-b BKG_FILE] [-p PIXELSIZE]
                                        [-d DISTANCE] [-t THRESHOLD] [--show]
                                        [--skip_plots] [--plot_file PLOT_FILE] [-s]
                                        [-w WINDOW] [-o OROOT] [--gau GAU]
                                        [--box BOX] [--model MODEL MODEL MODEL]
                                        img_file
    
    Calculate results for a collimated FRD test.
    
    positional arguments:
      img_file              File with far-field output image from a collimated input
    
    optional arguments:
      -h, --help            show this help message and exit
      -b BKG_FILE, --bkg_file BKG_FILE
                            File with only background flux (default: None)
      -p PIXELSIZE, --pixelsize PIXELSIZE
                            Size of the image camera pixels in mm. (default: 0.018)
      -d DISTANCE, --distance DISTANCE
                            Distance between the fiber output and the camera
                            detector (default: None)
      -t THRESHOLD, --threshold THRESHOLD
                            S/N threshold that sets the contour used to identify the
                            center of the output ring. (default: 1.5)
      --show                Display the QA plot in a window; do not write it to a
                            file. (default: False)
      --skip_plots          Skip the plot (default: False)
      --plot_file PLOT_FILE
                            Name of output plot file. If not provide, based on name
                            of input image file. (default: None)
      -s, --snr_img         If creating the QA plot, show the estimated S/N of the
                            data instead of the counts. (default: False)
      -w WINDOW, --window WINDOW
                            Limit the plotted image regions to this times the best-
                            fitting peak of the ring flux distribution. If None, the
                            full image is shown. (default: None)
      -o OROOT, --oroot OROOT
                            Directory for output files (default:
                            /Users/westfall/Work/packages/fobos/fiberlab/docs)
      --gau GAU             Smooth the image with a Gaussian kernel with this sigma
                            before analyzing the results. No smoothing is performed
                            by default. (default: None)
      --box BOX             Boxcar average the image before analyzing it (default:
                            None)
      --model MODEL MODEL MODEL
                            Modeling arguments: The step for the radial bin, the
                            smoothing filter window length, and the polynomial
                            order. The first value should be in units of pixels, mm,
                            or deg. Use pixels if --pixelsize and --distance are not
                            provided, mm if --pixelsize only is provided, and deg if
                            both are provided. If less than 0, no binning is
                            performed and the data is smoothed directly. The second
                            and last values are the window length and polynomial
                            order used by the Savitzky-Golay filter used to smooth
                            the data to create the "model". The window length should
                            be large when not smoothing. The polynomial order must
                            always be less than the window size, which must always
                            be less than the number of data points. (default: [-1,
                            301, 2])
    