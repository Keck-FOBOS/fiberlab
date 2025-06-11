.. code-block:: console

    $ fiberlab_fullcone_farfield -h
    usage: fiberlab_fullcone_farfield [-h] [--bkg_file BKG_FILE] [-p PIXELSIZE]
                                      [-d DISTANCE] [-l LEVEL] [-t THRESHOLD]
                                      [--show] [--skip_plots]
                                      [--plot_file PLOT_FILE] [-s] [--smooth]
                                      [-c BKG_CLIP BKG_CLIP BKG_CLIP]
                                      [-b BKG_LIM [BKG_LIM ...]] [--bkg_sig BKG_SIG]
                                      [-w WINDOW] [--box BOX] [-o OROOT]
                                      [--ofile OFILE]
                                      img_file
    
    Calculate results for a full-cone far-field test.
    
    positional arguments:
      img_file              File with far-field output image from a collimated input
    
    options:
      -h, --help            show this help message and exit
      --bkg_file BKG_FILE   File with only background flux (default: None)
      -p, --pixelsize PIXELSIZE
                            Size of the image camera pixels in mm. If None, analysis
                            done using pixel coordinates. (default: None)
      -d, --distance DISTANCE
                            Distance between the fiber output and the camera
                            detector (default: None)
      -l, --level LEVEL     The count level used to set the contour in the *binned*
                            image. I.e., if you use --box, this should be the value
                            after binning the image. If this is provided,
                            --threshold is ignored. (default: None)
      -t, --threshold THRESHOLD
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
      --smooth              Smooth the EE curve to limit interpolation errors
                            (default: False)
      -c, --bkg_clip BKG_CLIP BKG_CLIP BKG_CLIP
                            Clipping parameters used for setting the contour
                            threshold. Three numbers must be provided: (1) the
                            number of clipping iterations, (2) the lower rejection
                            threshold, and (3) the higher rejection threshold.
                            (default: [10.0, 100.0, 3.0])
      -b, --bkg_lim BKG_LIM [BKG_LIM ...]
                            One or multiple of the spot radius (as defined by the
                            fitted contour used to define spot center) to use for
                            the background estimate. If none, the background is
                            determined by first clipping the high valued pixels. If
                            one number is provided, all pixels above this radius are
                            used. If two numbers are provided, pixels between the
                            two radii are used. No more than 2 values can be given!
                            (default: None)
      --bkg_sig BKG_SIG     Sigma clipping level for the background measurement and
                            masking. (default: 3.0)
      -w, --window WINDOW   Limit the plotted image regions to this times the
                            fiducial radius of the far-field spot. If None, the full
                            image is shown. (default: None)
      --box BOX             Boxcar sum the image before analyzing it (default: None)
      -o, --oroot OROOT     Directory for output files (default:
                            /Users/westfall/Work/packages/fobos/fiberlab/docs)
      --ofile OFILE         Name of the file with discrete samples of the EE and
                            focal ratio. Not written if no file name is provided.
                            (default: None)
    