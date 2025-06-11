.. code-block:: console

    $ fiberlab_fullcone_throughput -h
    usage: fiberlab_fullcone_throughput [-h] [-b BKG_IMG] [-t THRESHOLD]
                                        inp_img out_img
    
    Calculate results for a full-cone far-field test.
    
    positional arguments:
      inp_img               File with an image of the input beam
      out_img               File with an image of the output beam
    
    options:
      -h, --help            show this help message and exit
      -b, --bkg_img BKG_IMG
                            File with only background flux (default: None)
      -t, --threshold THRESHOLD
                            S/N threshold that sets the contour used to identify the
                            center of the output ring. (default: 1.5)
    