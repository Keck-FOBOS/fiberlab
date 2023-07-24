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
        parser.add_argument('--plot_file', default=None, type=str,
                            help='Name of output plot file.  If not provide, based on name of '
                                 'input image file.')
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
        parser.add_argument('--box', default=None, type=int,
                            help='Boxcar average the image before analyzing it')
        parser.add_argument('-o', '--oroot', default=str(Path().resolve()), type=str,
                            help='Directory for output files')
        parser.add_argument('--ofile', default=None, type=str,
                            help='Name of the file with discrete samples of the EE and focal '
                                 'ratio.  Not written if no file name is provided.')
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
        from .. import contour

        img_file = Path(args.img_file).resolve()
        if not img_file.exists():
            raise FileNotFoundError(f'{img_file} does not exist!')
        bkg_file = None if args.bkg_file is None else Path(args.bkg_file).resolve()
        if bkg_file is not None and not bkg_file.exists():
            raise FileNotFoundError(f'{bkg_file} does not exist!')

        if args.oroot is None:
            oroot = img_file.parent
        else:
            oroot = Path(args.oroot).resolve()
        if not oroot.exists():
            oroot.mkdir(parents=True)

        # Analyze the image
        if args.skip_plots:
            plot_file = None
        elif args.show:
            plot_file = 'show'
        else:
            plot_file = oroot / (f'{img_file.with_suffix("").name}_qa.png'
                                    if args.plot_file is None else args.plot_file)

        print(f'Analyzing {img_file.name}')
        ee = fullcone.fullcone_farfield_output(img_file, bkg_file=bkg_file,
                                                  threshold=args.threshold,
                                                  pixelsize=args.pixelsize, plot_file=plot_file,
                                                  window=args.window, snr_img=args.snr_img, 
                                                  clip_iter=int(args.bkg_clip[0]),
                                                  sigma_lower=args.bkg_clip[1],
                                                  sigma_upper=args.bkg_clip[2],
                                                  bkg_lim=args.bkg_lim, bkg_lim_sig=args.bkg_sig,
                                                  box=args.box)

        # Raw values
        normalized_ee = ee.ee/ee.ee_norm
        if args.smooth:
            normalized_ee = contour.iterative_filter(normalized_ee, 301, 2)
        indx = numpy.where(numpy.diff(normalized_ee) < 0)[0]
        last = indx[0] if len(indx) > 0 else normalized_ee.size-1
        if args.bkg_lim is not None:
            last = min(last, numpy.where(ee.radius > args.bkg_lim[0]*ee.circ_p[2])[0][0])

        # "Model" values
        normalized_model_ee = ee.model_ee/ee.ee_norm
        indx = numpy.where(numpy.diff(normalized_model_ee) < 0)[0]
        model_last = indx[0] if len(indx) > 0 else normalized_model_ee.size-1
        if args.bkg_lim is not None:
            model_last = min(model_last,
                             numpy.where(ee.model_radius > args.bkg_lim[0]*ee.circ_p[2])[0][0])

        _pixelsize = args.pixelsize
        if args.box is not None:
            _pixelsize *= args.box

        try:
            ee90 = interpolate.interp1d(normalized_ee[:last],
                                        ee.radius[:last]*_pixelsize)(0.9)
        except:
            warnings.warn('Error interpolated raw EE data.')
            ee90 = None
            fratio90 = None
        else:
            fratio90 = None if args.distance is None else args.distance / 2 / ee90

        try:
            model_ee90 = interpolate.interp1d(normalized_model_ee[:model_last],
                                              ee.model_radius[:model_last]*_pixelsize)(0.9)
        except:
            warnings.warn('Error interpolated raw EE data.')
            model_ee90 = None
        else:
            model_fratio90 = None if args.distance is None else args.distance / 2 / model_ee90

        print('# Result from fullcone_farfield script')
        print(f'# Written: {time.strftime("%a %d %b %Y %H:%M:%S",time.localtime())}')
        print(f'# Image file: {img_file.name}')
        if bkg_file is not None:
            print(f'# Background image file: {bkg_file.name}')
        print(f'# Pixelsize: {args.pixelsize} mm')
        print(f'# Total Flux: {ee.ee_norm:.2f} ADU')
        print('# Radius at EE90 (mm): ' + ('Error' if ee90 is None else f'{ee90:.2f}'))
        print('# Model radius at EE90 (mm): ' + ('Error' if model_ee90 is None else f'{model_ee90:.2f}'))
        if args.distance is not None:
            print(f'# Distance from fiber output to image: {args.distance:.2f} mm')
            print('# f/# at EE90: ' + ('Error' if fratio90 is None else f'{fratio90:.2f}'))
            print('# Model of f/# at EE90: '
                  + ('Error' if model_fratio90 is None else f'{model_fratio90:.2f}'))

        if args.ofile is None:
            return

        # Main output file
        _ofile = Path(args.ofile).resolve()
        if _ofile.parent != oroot:
            _ofile = oroot / _ofile.name

        ee_sample, ee_r, ee_fratio, success \
                = fullcone.ee_to_fratio(ee.radius*_pixelsize, normalized_ee,
                                        distance=args.distance, smooth=False,
                                        bkg_lim=None if args.bkg_lim is None
                                                else args.bkg_lim[0]*ee.circ_p[2]*_pixelsize)

        results = 'Result from fobos_fullcone_farfield script\n' \
                  f'Written: {time.strftime("%a %d %b %Y %H:%M:%S",time.localtime())}\n\n' \
                  f'Top-level directory: {img_file.parent}\n\n' \
                  f'Image:         {img_file.name}\n'
        if bkg_file is not None:
            results += f'Background:    {bkg_file.name}\n'
        results += f'Pixelsize: {args.pixelsize} mm\n'
        results += '\n' if args.box is None else f'Boxcar: {args.box}\n\n'
        results += f'S/N Threshold: {args.threshold:.1f}\n' \
                   f'EE normalization (total flux): {ee.ee_norm:.2f} ADU\n'
        results += '\n' if args.distance is None else f'Distance: {args.distance:.2f} mm\n'
        header = results + '\nEE is the fractional inclosed energy\n' \
                 'R is the radius in mm at the detector plane\n' \
                 'f is the focal ratio assuming the provided distance\n\n' \
                 f'{"EE":>6} {"R":>6} {"f":>6}'
        numpy.savetxt(_ofile, numpy.column_stack((ee_sample, ee_r, ee_fratio)),
                      fmt=['%8.2f', '%6.2f', '%6.2f'], header=header)



