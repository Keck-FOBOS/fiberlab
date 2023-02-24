.. code-block:: console

    $ fiberlab_collimated_FRD -h
    usage: fiberlab_collimated_FRD [-h] [-p PIXELSIZE] [-s SEP] [-t THRESHOLD]
                                   [-r RING_BOX] [-o OROOT] [-f FILES] [-q]
                                   [--summary [SUMMARY]] [--na NA]
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
      -r RING_BOX, --ring_box RING_BOX
                            Limit the plotted image regions to this times the best-
                            fitting peak of the ring flux distribution. If None, the
                            full image is shown. (default: None)
      -o OROOT, --oroot OROOT
                            Directory for output files (default:
                            /Users/westfall/Work/packages/fobos/fiberlab/docs)
      -f FILES, --files FILES
                            Name of a file that provides 2 or 3 columns: (1) the
                            files to analyze, (2) the background image to use for
                            each file, and (3) the threshold to use for each file.
                            If this file is provided, any threshold is ignored and
                            the root directory is not trolled for all bg*, z*, and
                            a* files. The last column with the thresholds can be
                            omitted, which means the code will use the value
                            provided on the command line (or its default). (default:
                            None)
      -q, --no_qa           Skip making the individual QA plots for each image.
                            (default: True)
      --summary [SUMMARY]   Produce a summary plot showing the ring width as a
                            function of input angle/f-ratio. The argument given must
                            be the name of the file for the output plot; if no
                            argument is given, the plot is only shown on screen.
                            (default: None)
      --na NA               The numerical aperture of the fiber. Only used when
                            producing the summary plot. (default: None)
    