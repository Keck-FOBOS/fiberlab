"""
Script that produces results for a collimated FRD test.
""" 

from . import scriptbase


class FullConeEEFRatio(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        from pathlib import Path
        from ..fullcone import default_threshold
        parser = super().get_parser(description='Calculate results for a full-cone far-field test.',
                                    width=width)
        parser.add_argument('root', default=str(Path().resolve()), type=str,
                            help='Directory with output files.')
        parser.add_argument('ofile', default='fullcone_ee_fratio.txt', type=str,
                            help='Output file for measurements')
        parser.add_argument('-i', '--images', nargs=2, default=None, type=str,
                            help='Two files with bench images taken at separate distances from '
                                 'the fiber output.  If None, script tries to find them in the '
                                 'provided directory.')
        parser.add_argument('-f', '--files', default=None, type=str,
                            help='Name of a file that provides 2 or 3 columns: (1) the '
                                 'files to analyze, (2) the background image to use for each '
                                 'file, and (3) the threshold to use for each file.  If this '
                                 'file is provided, any threshold is ignored and the root '
                                 'directory is not trolled for all bg* and z* files.  The '
                                 'last column with the thresholds can be omitted, which means '
                                 'the code will use the value provided on the command line (or '
                                 'its default).')
        parser.add_argument('-p', '--pixelsize', default=0.018, type=float,
                            help='Size of the image camera pixels in mm.')
        parser.add_argument('-s', '--sep', default=3.82, type=float,
                            help='Known separation (in mm) between the "z" images used to '
                                 'calculate the distance from the fiber output to the main '
                                 'imaging position.')
        parser.add_argument('-t', '--threshold', default=default_threshold(), type=float,
                            help='S/N threshold that sets the contour used to identify the center '
                                 'of the output ring.')
        parser.add_argument('--smooth', default=False, action='store_true',
                            help='Smooth the EE curve to limit interpolation errors')
        # TODO: Add smoothing option?
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
        parser.add_argument('-o', '--oroot', default=str(Path().resolve()), type=str,
                            help='Directory for output files')
        parser.add_argument('--skip_plots', default=False, action='store_true',
                            help='Only create the output data file and skip the plots.')
        parser.add_argument('--box', default=None, type=int,
                            help='Boxcar average the image before analyzing it')
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
        from .. import io

        # Set the root path
        root = Path(args.root).resolve()
        if not root.exists():
            raise FileNotFoundError(f'{root} is not a valid directory.')

        oroot = Path(args.oroot).resolve()
        if not oroot.exists():
            oroot.mkdir(parents=True)

        if args.images is None:
            # Get files
            z0, z0_bg, z0_thresh, z1, z1_bg, z1_thresh \
                    = io.gather_fullcone_file_list(root, par=args.files, threshold=args.threshold)
        else:
            z0 = root / args.images[0]
            if not z0.exists():
                raise FileNotFoundError(f'{z0.name} does not exist in {root}!')
            z0_bg = None
            z0_thresh = args.threshold
            z1 = root / args.images[1]
            if not z1.exists():
                raise FileNotFoundError(f'{z1.name} does not exist in {root}!')
            z1_bg = None
            z1_thresh = args.threshold

        # Analyze the two baseline images
        plot_file = None if args.skip_plots else oroot / f'{z0.with_suffix("").name}_qa.png'
        print(f'Analyzing {z0.name}')
        z0_ee = fullcone.fullcone_farfield_output(z0, bkg_file=z0_bg, threshold=z0_thresh,
                                                  pixelsize=args.pixelsize, plot_file=plot_file,
                                                  window=args.window,
                                                  clip_iter=int(args.bkg_clip[0]),
                                                  sigma_lower=args.bkg_clip[1],
                                                  sigma_upper=args.bkg_clip[2],
                                                  bkg_lim=args.bkg_lim, bkg_lim_sig=args.bkg_sig,
                                                  box=args.box)

        plot_file = None if args.skip_plots else oroot / f'{z1.with_suffix("").name}_qa.png'
        print(f'Analyzing {z1.name}')
        z1_ee = fullcone.fullcone_farfield_output(z1, bkg_file=z1_bg, threshold=z1_thresh,
                                                  pixelsize=args.pixelsize, plot_file=plot_file,
                                                  window=args.window,
                                                  clip_iter=int(args.bkg_clip[0]),
                                                  sigma_lower=args.bkg_clip[1],
                                                  sigma_upper=args.bkg_clip[2],
                                                  bkg_lim=args.bkg_lim, bkg_lim_sig=args.bkg_sig,
                                                  box=args.box)
        pixelsize = args.pixelsize
        if args.box is not None:
            pixelsize *= args.box

        # Use the raw data to do the distance and focal ratio calculation
        ee_sample, ee_r_z0, ee_r_z1, z0_distance, z1_distance, \
            med_z0_distance, med_z1_distance, ee_fratio_z0, ee_fratio_z1, success \
                = fullcone.calculate_fratio(z0_ee.radius*pixelsize, z0_ee.ee/z0_ee.ee_norm,
                                            z1_ee.radius*pixelsize, z1_ee.ee/z1_ee.ee_norm,
                                            args.sep, smooth=args.smooth,
                                            z0_bkg_lim=None if args.bkg_lim is None 
                                                else args.bkg_lim[0]*z0_ee.circ_p[2]*pixelsize,
                                            z1_bkg_lim=None if args.bkg_lim is None 
                                                else args.bkg_lim[0]*z1_ee.circ_p[2]*pixelsize)

        # Use the "model" data to do the distance and focal ratio calculation.
        # Model data is not smoothed.
        ee_sample, model_ee_r_z0, model_ee_r_z1, model_z0_distance, model_z1_distance, \
            model_med_z0_distance, model_med_z1_distance, model_ee_fratio_z0, model_ee_fratio_z1, \
            model_success \
                = fullcone.calculate_fratio(z0_ee.model_radius*pixelsize,
                                            z0_ee.model_ee/z0_ee.ee_norm,
                                            z1_ee.model_radius*pixelsize,
                                            z1_ee.model_ee/z1_ee.ee_norm,
                                            args.sep,
                                            z0_bkg_lim=None if args.bkg_lim is None 
                                                else args.bkg_lim[0]*z0_ee.circ_p[2]*pixelsize,
                                            z1_bkg_lim=None if args.bkg_lim is None 
                                                else args.bkg_lim[0]*z1_ee.circ_p[2]*pixelsize)

        # Main output file
        _ofile = Path(args.ofile).resolve()
        if _ofile.parent != oroot:
            _ofile = oroot / _ofile.name

        results = 'Result from fobos_fullcone_ee_fratio script\n' \
                  f'Written: {time.strftime("%a %d %b %Y %H:%M:%S",time.localtime())}\n\n' \
                  f'Top-level directory: {root}\n\n' \
                  f'Pixelsize: {args.pixelsize} mm\n'
        if args.box is None:
            results += '\n'
        else:
            results += f'Boxcar: {args.box}\n\n'
        results += f'Z0 Image:         {z0.name}\n'
        if z0_bg is not None:
            results += f'Z0 Background:    {z0_bg.name}\n'
        results += f'Z0 S/N Threshold: {z0_thresh:.1f}\n' \
                   f'Z0 EE normalization (total flux): {z0_ee.ee_norm:.4e} ADU\n' \
                   f'Median Z0 distance (0.2 < EE < 0.9) RAW: {med_z0_distance:.2f} mm\n' \
                   f'Median Z0 distance (0.2 < EE < 0.9) MODEL: {model_med_z0_distance:.2f} mm\n\n' \
                   f'Z1 Image:         {z1.name}\n'
        if z1_bg is not None:
            results += f'Z1 Background:    {z1_bg.name}\n'
        results += f'Z1 S/N Threshold: {z1_thresh:.1f}\n' \
                   f'Z1 EE normalization (total flux): {z1_ee.ee_norm:.4e} ADU\n' \
                   f'Median Z1 distance (0.2 < EE < 0.9) RAW: {med_z1_distance:.2f} mm\n' \
                   f'Median Z1 distance (0.2 < EE < 0.9) MODEL: {model_med_z1_distance:.2f} mm\n\n'
        header = results + 'EE is the fractional inclosed energy\n' \
                 'R0 is the radius in mm at the detector plane at the closest (z0) image\n' \
                 'R1 is the radius in mm at the detector plane at the farthest (z1) image\n' \
                 'F0 is the distance in mm to the closest (z0) image using R0\n' \
                 'F1 is the distance in mm to the farthest (z1) image using R1\n' \
                 'f0 is the focal ratio assuming the median z0 distance\n' \
                 'f1 is the focal ratio assuming the median z1 distance\n\n' \
                 'MR0 is the (model) radius in mm at the detector plane at the closest (z0) image\n' \
                 'MR1 is the (model) radius in mm at the detector plane at the farthest (z1) image\n' \
                 'MF0 is the (model) distance in mm to the closest (z0) image using R0\n' \
                 'MF1 is the (model) distance in mm to the farthest (z1) image using R1\n' \
                 'Mf0 is the (model) focal ratio assuming the median z0 distance\n' \
                 'Mf1 is the (model) focal ratio assuming the median z1 distance\n\n' \
                 f'{"EE":>6} {"R0":>6} {"R1":>6} {"F0":>6} {"F1":>6} {"f0":>6} {"f1":>6} ' \
                 f'{"MR0":>6} {"MR1":>6} {"MF0":>6} {"MF1":>6} {"Mf0":>6} {"Mf1":>6}'
        numpy.savetxt(_ofile, numpy.column_stack((ee_sample,
                                                  ee_r_z0, ee_r_z1, z0_distance,
                                                  z1_distance, ee_fratio_z0, ee_fratio_z1,
                                                  model_ee_r_z0, model_ee_r_z1,
                                                  model_z0_distance, model_z1_distance,
                                                  model_ee_fratio_z0, model_ee_fratio_z1)),
                      fmt=['%8.2f', '%6.2f', '%6.2f', '%6.2f', '%6.2f', '%6.2f', '%6.2f',
                                    '%6.2f', '%6.2f', '%6.2f', '%6.2f', '%6.2f', '%6.2f'],
                      header=header)

        # Also print the results to the screen
        print(results)

