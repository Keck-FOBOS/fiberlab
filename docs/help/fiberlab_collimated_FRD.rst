.. code-block:: console

    $ fiberlab_collimated_FRD -h
    usage: fiberlab_collimated_FRD [-h] [-p PIXELSIZE] [-s SEP] [-t THRESHOLD]
                                   [-w WINDOW] [-o OROOT] [-f FILES] [-q]
                                   [--gau GAU] [--box BOX]
                                   [--model MODEL MODEL MODEL] [--summary [SUMMARY]]
                                   [--na NA]
                                   root ofile
    
    Calculate results for a collimated FRD test.
    
    positional arguments:
      root                  Directory with output files.
      ofile                 Output file for measurements
    
    optional arguments:
      -h, --help            show this help message and exit
      -p PIXELSIZE, --pixelsize PIXELSIZE
                            Size of the image camera pixels in mm. (default: 0.018)
      -s SEP, --sep SEP     Known separation (in mm) between the "z" images used to
                            calculate the distance from the fiber output to the main
                            imaging position. (default: 3.81)
      -t THRESHOLD, --threshold THRESHOLD
                            S/N threshold that sets the contour used to identify the
                            center of the output ring. (default: 1.5)
      -w WINDOW, --window WINDOW
                            Limit the plotted image regions to this times the best-
                            fitting peak of the ring flux distribution. If None, the
                            full image is shown. (default: None)
      -o OROOT, --oroot OROOT
                            Directory for output files (default:
                            /Users/westfall/Work/packages/fobos/fiberlab/docs)
      -f FILES, --files FILES
                            Name of a file that provides 2, 3, or 4 columns: (1) the
                            files to analyze, (2) the background image to use for
                            each file, (3) the threshold to use for each file, and
                            (4) the image designation. If this file is provided, any
                            threshold is ignored and the root directory is not
                            trolled for all bg*, z*, and a* files. The last column
                            with the thresholds can be omitted, which means the code
                            will use the value provided on the command line (or its
                            default). (default: None)
      -q, --no_qa           Skip making the individual QA plots for each image.
                            (default: True)
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
      --summary [SUMMARY]   Produce a summary plot showing the ring width as a
                            function of input angle/f-ratio. The argument given must
                            be the name of the file for the output plot; if no
                            argument is given, the plot is only shown on screen.
                            (default: None)
      --na NA               The numerical aperture of the fiber. Only used when
                            producing the summary plot. (default: None)
    