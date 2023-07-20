"""
Script that produces results for a collimated FRD test.
""" 

from . import scriptbase


class CollimatedFarField(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        from pathlib import Path
        parser = super().get_parser(description='Calculate results for a collimated FRD test.',
                                    width=width)
        parser.add_argument('img_file', default=None, type=str,
                            help='File with far-field output image from a collimated input')
        parser.add_argument('-b', '--bkg_file', default=None, type=str,
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
        parser.add_argument('--plot_file', default=None, type=str,
                            help='Name of output plot file.  If not provide, based on name of '
                                 'input image file.')
        parser.add_argument('-s', '--snr_img', default=False, action='store_true',
                            help='If creating the QA plot, show the estimated S/N of the data '
                                 'instead of the counts.')
        parser.add_argument('-w', '--window', default=None, type=float,
                            help='Limit the plotted image regions to this times the best-fitting '
                                 'peak of the ring flux distribution.  If None, the full image '
                                 'is shown.')
        parser.add_argument('--gau', default=None, type=float,
                            help='Smooth the image with a Gaussian kernel with this sigma before '
                                 'analyzing the results.  No smoothing is performed by default.')
        parser.add_argument('--box', default=None, type=int,
                            help='Boxcar average the image before analyzing it')

        parser.add_argument('--model', default=[-1, 301, 2], nargs=3, type=float,
                            help='Modeling arguments: The step for the radial bin, the smoothing '
                                 'filter window length, and the polynomial order.  The first '
                                 'value should be in units of pixels, mm, or deg.  Use pixels '
                                 'if --pixelsize and --distance are not provided, mm if '
                                 '--pixelsize only is provided, and deg if both are provided.  If '
                                 'less than 0, no binning is performed and the data is smoothed '
                                 'directly.  The second and last values are the window length '
                                 'and polynomial order used by the Savitzky-Golay filter used to '
                                 'smooth the data to create the "model".  The window length '
                                 'should be large when not smoothing.  The polynomial order must '
                                 'always be less than the window size, which must always be less '
                                 'than the number of data points.')
        return parser

    @staticmethod
    def main(args):

        import time
        from pathlib import Path
        import warnings

        from IPython import embed

        from .. import collimated

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
            plot_file = oroot / (f'{img_file.with_suffix("").name}_qa.png'
                                    if args.plot_file is None else args.plot_file)

        print(f'Analyzing {img_file.name}')
        dr = None if args.model[0] < 0 else args.model[0]
        savgol = tuple([int(m) for m in args.model[1:]])
        rad, peak, fwhm = collimated.collimated_farfield_output(img_file, bkg_file=bkg_file,
                                                                threshold=args.threshold,
                                                                pixelsize=args.pixelsize,
                                                                distance=args.distance,
                                                                plot_file=plot_file,
                                                                snr_img=args.snr_img,
                                                                window=args.window,
                                                                gau=args.gau, box=args.box,
                                                                dr=dr, savgol=savgol)
        print('# Result from fobos_collimated_farfield script')
        print(f'# Written: {time.strftime("%a %d %b %Y %H:%M:%S",time.localtime())}')
        print(f'# Image file: {img_file.name}')
        print(f'# Pixelsize: {args.pixelsize} mm')
        if args.distance is not None:
            print(f'# Distance from fiber output to image: {args.distance:.2f} mm')
        print(f'# Ring radius:    {rad:.2f}')
        print(f'# Ring peak flux: {peak:.2f}')
        print(f'# Ring FWHM:      {fwhm:.2f}')



