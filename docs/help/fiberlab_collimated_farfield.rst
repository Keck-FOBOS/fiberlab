.. code-block:: console

    $ fiberlab_collimated_farfield -h
    usage: fiberlab_collimated_farfield [-h] [-b BKG_FILE] [-p PIXELSIZE]
                                        [-d DISTANCE] [-t THRESHOLD] [-w] [-s]
                                        [-r RING_BOX]
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
      -w, --window          Display the QA plot in a window; do not write it to a
                            file. (default: False)
      -s, --snr_img         If creating the QA plot, show the estimated S/N of the
                            data instead of the counts. (default: False)
      -r RING_BOX, --ring_box RING_BOX
                            Limit the plotted image regions to this times the best-
                            fitting peak of the ring flux distribution. If None, the
                            full image is shown. (default: None)
    