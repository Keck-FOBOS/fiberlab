.. code-block:: console

    $ fiberlab_browse_images -h
    usage: fiberlab_browse_images [-h] [-i IMAGE] [-s SEARCH] [-e EXT]
                                  [-z ZLIM ZLIM]
                                  root
    
    Browse images in a directory
    
    positional arguments:
      root                  Directory with output files.
    
    options:
      -h, --help            show this help message and exit
      -i, --image IMAGE     Name of single file to show. If provided, -s and -e
                            arguments are ignored. (default: None)
      -s, --search SEARCH   Search string for image names (default: None)
      -e, --ext EXT         Image extension (default: .fit)
      -z, --zlim ZLIM ZLIM  The upper and lower values to use for the image plot
                            limits. Default is to plot +/- 5 sigma around the image
                            mean. (default: None)
    