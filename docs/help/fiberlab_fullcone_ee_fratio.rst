.. code-block:: console

    $ fiberlab_fullcone_ee_fratio -h
    usage: fiberlab_fullcone_ee_fratio [-h] [-i IMAGES IMAGES] [-f FILES]
                                       [-p PIXELSIZE] [-s SEP] [-t THRESHOLD]
                                       [--smooth] [-c BKG_CLIP BKG_CLIP BKG_CLIP]
                                       [-b BKG_LIM [BKG_LIM ...]]
                                       [--bkg_sig BKG_SIG] [-w WINDOW] [-o OROOT]
                                       [--skip_plots]
                                       root ofile
    
    Calculate results for a full-cone far-field test.
    
    positional arguments:
      root                  Directory with output files.
      ofile                 Output file for measurements
    
    optional arguments:
      -h, --help            show this help message and exit
      -i IMAGES IMAGES, --images IMAGES IMAGES
                            Two files with bench images taken at separate distances
                            from the fiber output. If None, script tries to find
                            them in the provided directory. (default: None)
      -f FILES, --files FILES
                            Name of a file that provides 2 or 3 columns: (1) the
                            files to analyze, (2) the background image to use for
                            each file, and (3) the threshold to use for each file.
                            If this file is provided, any threshold is ignored and
                            the root directory is not trolled for all bg* and z*
                            files. The last column with the thresholds can be
                            omitted, which means the code will use the value
                            provided on the command line (or its default). (default:
                            None)
      -p PIXELSIZE, --pixelsize PIXELSIZE
                            Size of the image camera pixels in mm. (default: 0.018)
      -s SEP, --sep SEP     Known separation (in mm) between the "z" images used to
                            calculate the distance from the fiber output to the main
                            imaging position. (default: 3.82)
      -t THRESHOLD, --threshold THRESHOLD
                            S/N threshold that sets the contour used to identify the
                            center of the output ring. (default: 3.0)
      --smooth              Smooth the EE curve to limit interpolation errors
                            (default: False)
      -c BKG_CLIP BKG_CLIP BKG_CLIP, --bkg_clip BKG_CLIP BKG_CLIP BKG_CLIP
                            Clipping parameters used for setting the contour
                            threshold. Three numbers must be provided: (1) the
                            number of clipping iterations, (2) the lower rejection
                            threshold, and (3) the higher rejection threshold.
                            (default: [10.0, 100.0, 3.0])
      -b BKG_LIM [BKG_LIM ...], --bkg_lim BKG_LIM [BKG_LIM ...]
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
      -w WINDOW, --window WINDOW
                            Limit the plotted image regions to this times the
                            fiducial radius of the far-field spot. If None, the full
                            image is shown. (default: None)
      -o OROOT, --oroot OROOT
                            Directory for output files (default:
                            /Users/westfall/Work/packages/fobos/fiberlab/docs)
      --skip_plots          Only create the output data file and skip the plots.
                            (default: False)
    