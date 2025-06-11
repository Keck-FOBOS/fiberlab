.. code-block:: console

    $ fiberlab_farfield_inspector -h
    usage: fiberlab_farfield_inspector [-h] [--bkg_file BKG_FILE] [-p PIXELSIZE]
                                       [-d DISTANCE] [--smooth] [--box BOX]
                                       img_file
    
    Interactively inspect a far-field image
    
    positional arguments:
      img_file              File with far-field output image from a collimated input
    
    options:
      -h, --help            show this help message and exit
      --bkg_file BKG_FILE   File with only background flux (default: None)
      -p, --pixelsize PIXELSIZE
                            Size of the image camera pixels in mm. (default: None)
      -d, --distance DISTANCE
                            Distance between the fiber output and the camera
                            detector (default: None)
      --smooth              Smooth the EE curve to limit interpolation errors
                            (default: False)
      --box BOX             Boxcar average the image before analyzing it (default:
                            None)
    