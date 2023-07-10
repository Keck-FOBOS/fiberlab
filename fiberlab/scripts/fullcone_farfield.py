"""
Script that produces results for a collimated FRD test.
""" 

from . import scriptbase


class FullConeFarField(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        from pathlib import Path
        parser = super().get_parser(description='Calculate results for a full-cone far-field test.',
                                    width=width)
        parser.add_argument('img_file', default=None, type=str,
                            help='File with far-field output image from a collimated input')
        parser.add_argument('--bkg_file', default=None, type=str,
                            help='File with only background flux')
        parser.add_argument('-p', '--pixelsize', default=0.018, type=float,
                            help='Size of the image camera pixels in mm.')
        parser.add_argument('-d', '--distance', default=None, type=float,
                            help='Distance between the fiber output and the camera detector')
        parser.add_argument('-t', '--threshold', default=1.5, type=float,
                            help='S/N threshold that sets the contour used to identify the center '
                                 'of the output ring.')
        parser.add_argument('--show', default=False, action='store_true',
                            help='Display the QA plot in a window; do not write it to a file.')
        parser.add_argument('--skip_plots', default=False, action='store_true',
                            help='Skip the plot')
        parser.add_argument('-s', '--snr_img', default=False, action='store_true',
                            help='If creating the QA plot, show the estimated S/N of the data '
                                 'instead of the counts.')
        parser.add_argument('--smooth', default=False, action='store_true',
                            help='Smooth the EE curve to limit interpolation errors')
        parser.add_argument('-c', '--bkg_clip', nargs=3, default=[10., 100., 3.], type=float,
                            help='Clipping parameters used for setting the contour threshold.  '
                                 'Three numbers must be provided: (1) the number of clipping '
                                 'iterations, (2) the lower rejection threshold, and (3) the '
                                 'higher rejection threshold.')
        parser.add_argument('-b', '--bkg_lim', nargs='+', default=None, type=float,
                            help='One or multiple of the spot radius (as defined by the fitted '
                                 'contour used to define spot center) to use for the background '
                                 'estimate.  If none, the background is determined by first '
                                 'clipping the high valued pixels.  If one number is provided, '
                                 'all pixels above this radius are used.  If two numbers are '
                                 'provided, pixels between the two radii are used.  No more than '
                                 '2 values can be given!')
        parser.add_argument('--bkg_sig', default=3., type=float,
                            help='Sigma clipping level for the background measurement and masking.')
        parser.add_argument('-w', '--window', default=None, type=float,
                            help='Limit the plotted image regions to this times the fiducial '
                                 'radius of the far-field spot. If None, the full image '
                                 'is shown.')
        return parser

    @staticmethod
    def main(args):

        import time
        from pathlib import Path
        import warnings

        from IPython import embed

        import numpy
        from scipy import interpolate

        from .. import fullcone

        img_file = Path(args.img_file).resolve()
        if not img_file.exists():
            raise FileNotFoundError(f'{img_file} does not exist!')
        bkg_file = None if args.bkg_file is None else Path(args.bkg_file).resolve()
        if bkg_file is not None and not bkg_file.exists():
            raise FileNotFoundError(f'{bkg_file} does not exist!')

        # Analyze the image
        if args.skip_plots:
            plot_file = None
        elif args.show:
            plot_file = 'show'
        else:
            plot_file = img_file.parent / f'{img_file.with_suffix("").name}_qa.png'
        print(f'Analyzing {img_file.name}')
        z0_ee = fullcone.fullcone_farfield_output(img_file, bkg_file=bkg_file,
                                                  threshold=args.threshold,
                                                  pixelsize=args.pixelsize, plot_file=plot_file,
                                                  window=args.window, snr_img=args.snr_img, 
                                                  clip_iter=int(args.bkg_clip[0]),
                                                  sigma_lower=args.bkg_clip[1],
                                                  sigma_upper=args.bkg_clip[2],
                                                  bkg_lim=args.bkg_lim, bkg_lim_sig=args.bkg_sig)

        z0_ee_norm = z0_ee.ee/z0_ee.ee_norm
        if args.smooth:
            z0_ee_norm = contour.iterative_filter(z0_ee_norm, 301, 2)

        z0_last = numpy.where(numpy.diff(z0_ee_norm) < 0)[0][0]
        if args.bkg_lim is not None:
            z0_last = min(z0_last, numpy.where(z0_ee.radius > args.bkg_lim[0]*z0_ee.circ_p[2])[0][0])
#        print(z0_last, z0_ee.radius[z0_last]*args.pixelsize)

        ee90 = interpolate.interp1d(z0_ee_norm[:z0_last],
                                    z0_ee.radius[:z0_last]*args.pixelsize)(0.9)
        fratio90 = None if args.distance is None else args.distance / 2 / ee90

        print('# Result from fullcone_farfield script')
        print(f'# Written: {time.strftime("%a %d %b %Y %H:%M:%S",time.localtime())}')
        print(f'# Image file: {img_file.name}')
        print(f'# Pixelsize: {args.pixelsize} mm')
        print(f'# Total Flux: {z0_ee.ee_norm:.2f}')
        print(f'# EE90: {ee90:.2f}')
        if args.distance is not None:
            print(f'# Distance from fiber output to image: {args.distance:.2f} mm')
            print(f'# f/# at EE90: {fratio90:.2f}')



