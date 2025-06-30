"""
Script that produces results for a collimated FRD test.
""" 
from . import scriptbase

class CollimatedFRD(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        from pathlib import Path
        from ..collimated import default_threshold
        parser = super().get_parser(description='Calculate results for a collimated FRD test.',
                                    width=width)
        parser.add_argument('root', default=str(Path().resolve()), type=str,
                            help='Directory with output files.')
        parser.add_argument('ofile', default='collimated_FRD.txt', type=str,
                            help='Output file for measurements')
        parser.add_argument('-p', '--pixelsize', default=0.018, type=float,
                            help='Size of the image camera pixels in mm.')
        parser.add_argument('-s', '--sep', default=3.81, type=float,
                            help='Known separation (in mm) between the "z" images used to '
                                 'calculate the distance from the fiber output to the main '
                                 'imaging position.')
        parser.add_argument('-t', '--threshold', default=default_threshold(), type=float,
                            help='S/N threshold that sets the contour used to identify the center '
                                 'of the output ring.')
        parser.add_argument('-d', '--dist_ref', default='z1', type=str,
                            help='Image used for distance reference.  Must be z0 (for the close '
                                 'image) or z1 (for the far image)')
        parser.add_argument('-w', '--window', default=None, type=float,
                            help='Limit the plotted image regions to this times the best-fitting '
                                 'peak of the ring flux distribution.  If None, the full image '
                                 'is shown.')
        parser.add_argument('-o', '--oroot', default=str(Path().resolve()), type=str,
                            help='Directory for output files')
        parser.add_argument('-f', '--files', default=None, type=str,
                            help='Name of a file that provides 2, 3, or 4 columns: (1) the '
                                 'files to analyze, (2) the background image to use for each '
                                 'file, (3) the threshold to use for each file, and (4) the '
                                 'image designation.  If this '
                                 'file is provided, any threshold is ignored and the root '
                                 'directory is not trolled for all bg*, z*, and a* files.  The '
                                 'last column with the thresholds can be omitted, which means '
                                 'the code will use the value provided on the command line (or '
                                 'its default).')
        parser.add_argument('-q', '--no_qa', dest='qa', default=True, action='store_false',
                            help='Skip making the individual QA plots for each image.')

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

        parser.add_argument('--summary', nargs='?', const='', default=None,
                            help='Produce a summary plot showing the ring width as a function of '
                                 'input angle/f-ratio.  The argument given must be the name of '
                                 'the file for the output plot; if no argument is given, the plot '
                                 'is only shown on screen.')
        parser.add_argument('--na', default=None, type=float,
                            help='The numerical aperture of the fiber.  Only used when producing '
                                 'the summary plot.')
        parser.add_argument('--fratio', default=None, type=float,
                            help='Interpolate the observations to predict the output f-ratio at '
                                 'this value of the input f-ratio.  If not provided, no '
                                 'interpolation is performed.')
        return parser

    @staticmethod
    def main(args):

        import time
        from pathlib import Path
        import warnings

        from IPython import embed

        import numpy

        from .. import collimated
        from .. import io
        from .. import plot

        if args.dist_ref not in ['z0', 'z1']:
            raise ValueError(f'Distance reference must be z0 or z1, not {args.dist_ref}.')

        # Set the root path
        root = Path(args.root).resolve()
        if not root.exists():
            raise FileNotFoundError(f'{root} is not a valid directory.')

        oroot = Path(args.oroot).resolve()
        if not oroot.exists():
            oroot.mkdir(parents=True)

        # Get files
        z0, z0_bg, z0_thresh, z1, z1_bg, z1_thresh, afiles, a_bg, a_thresh \
                = io.gather_collimated_file_list(root, par=args.files, threshold=args.threshold)

        # Parameters used to generate the model ring profile
        dr = None if args.model[0] < 0 else args.model[0]
        savgol = tuple([int(m) for m in args.model[1:]])

        # Analyze the two baseline images
        plot_file = oroot / f'{z0.with_suffix("").name}_qa.png' if args.qa else None
        print(f'Analyzing {z0.name}')
        z0_rad, z0_peak, z0_fwhm \
                = collimated.collimated_farfield_output(z0, bkg_file=z0_bg, threshold=z0_thresh,
                                                        pixelsize=args.pixelsize,
                                                        plot_file=plot_file, window=args.window,
                                                        gau=args.gau, box=args.box, dr=dr,
                                                        savgol=savgol)
        plot_file = oroot / f'{z1.with_suffix("").name}_qa.png' if args.qa else None
        print(f'Analyzing {z1.name}')
        z1_rad, z1_peak, z1_fwhm \
                = collimated.collimated_farfield_output(z1, bkg_file=z1_bg, threshold=z1_thresh,
                                                        pixelsize=args.pixelsize,
                                                        plot_file=plot_file, window=args.window,
                                                        gau=args.gau, box=args.box, dr=dr,
                                                        savgol=savgol)

        # Use the known distance between the two z images to get the distance
        # between the fiber output and z1.
        if z0_rad > z1_rad:
            warnings.warn('z0 image should be closer to fiber output.  However, comparison '
                          'between z0 and z1 images show that the z1 image was closer.  Flipping '
                          'the file names for the analysis.')
            # Swap all the z* objects
            z0, z1 = z1, z0
            z0_bg, z1_bg = z1_bg, z0_bg
            z0_thresh, z1_thresh = z1_thresh, z0_thresh
            z0_rad, z1_rad = z1_rad, z0_rad
            z0_peak, z1_peak = z1_peak, z0_peak
            z0_fwhm, z1_fwhm = z1_fwhm, z0_fwhm

        z0_distance = args.sep/(z1_rad/z0_rad-1)
        z1_distance = args.sep/(1-z0_rad/z1_rad)

        ref_distance = z0_distance if args.dist_ref == 'z0' else z1_distance

        if len(afiles) > 0:
            nangle = len(afiles)
            a_rad = numpy.zeros(nangle, dtype=float)
            a_peak = numpy.zeros(nangle, dtype=float)
            a_fwhm = numpy.zeros(nangle, dtype=float)
            for i in range(nangle):
                print(f'Analyzing {afiles[i].name}')
                plot_file = oroot / f'{afiles[i].with_suffix("").name}_qa.png' if args.qa else None
                a_rad[i], a_peak[i], a_fwhm[i] \
                        = collimated.collimated_farfield_output(afiles[i], bkg_file=a_bg[i],
                                                                threshold=a_thresh[i],
                                                                pixelsize=args.pixelsize,
                                                                distance=ref_distance,
                                                                plot_file=plot_file,
                                                                window=args.window,
                                                                gau=args.gau, box=args.box, dr=dr,
                                                                savgol=savgol)
        else:
            a_rad = a_peak = a_fwhm = None

        in_fratio = 1 / 2 / numpy.tan(numpy.radians(a_rad))
        out_fratio = 1 / 2 / numpy.tan(numpy.radians(a_rad + a_fwhm/2))

        # TODO:
        #   - Save sigma and level
        #   - Create plot that shows results

        # Main output file
        _ofile = Path(args.ofile).resolve()
        if _ofile.parent != oroot:
            _ofile = oroot / _ofile.name
        with open(_ofile, 'w') as f:
            f.write('# Result from fobos_collimated_FRD script\n')
            f.write(f'# Written: {time.strftime("%a %d %b %Y %H:%M:%S",time.localtime())}\n#\n')
            f.write(f'# Top-level directory: {root}\n#\n')
            f.write(f'# Pixelsize: {args.pixelsize} mm\n')
            f.write(f'# Z image separation: {args.sep} mm\n#\n')
            f.write(f'# Z0 image:\n')
            f.write(f'#     File:           {z0.name}\n')
            f.write(f'#     Background:     {z0_bg.name}\n')
            f.write(f'#     S/N Threshold:  {z0_thresh:.1f}\n')
            f.write(f'#     Ring radius:    {z0_rad:.2f}\n')
            f.write(f'#     Ring peak flux: {z0_peak:.2f}\n')
            f.write(f'#     Ring FWHM:      {z0_fwhm:.2f}\n')
            f.write(f'# Z1 image:\n')
            f.write(f'#     File:           {z1.name}\n')
            f.write(f'#     Background:     {z1_bg.name}\n')
            f.write(f'#     S/N Threshold:  {z1_thresh:.1f}\n')
            f.write(f'#     Ring radius:    {z1_rad:.2f}\n')
            f.write(f'#     Ring peak flux: {z1_peak:.2f}\n')
            f.write(f'#     Ring FWHM:      {z1_fwhm:.2f}\n#\n')
            f.write(f'# Distance from fiber output to z0 image: {z0_distance:.2f} mm\n#\n')
            f.write(f'# Distance from fiber output to z1 image: {z1_distance:.2f} mm\n#\n')
            f.write(f'# Reference distance for angle sweep is: {ref_distance:.2f} mm\n#\n')
            if a_rad is None:
                f.write(f'# No angle-sweep images to analyze\n')
            else:
                f.write('# Angle sweep results\n')
                f.write('# Radius and FWHM are in degrees.\n')
                if args.na is not None:
                    f.write(f'# Fastest f-ratio expected for NA={args.na}: {1/2/args.na:.2f}\n')
                if args.fratio is None:
                    f.write('#\n')
                else:
                    f.write(f'# Expected output f-ratio for an input f-ratio of {args.fratio}: '
                            f'{numpy.interp([args.fratio], in_fratio, out_fratio)[0]:.3f}\n#\n')

                f.write(f'# {"FILE":>15} {"BKG":>15} {"THRESH":>6} {"PEAK":>10} {"RAD":>8} '
                        f'{"FWHM":>8} {"F_IN":>6} {"F_OUT":>6}\n')
                for i in range(nangle):
                    f.write(f'  {afiles[i].name:>15} {a_bg[i].name:>15} {a_thresh[i]:6.1f} '
                            f'{a_peak[i]:10.2e} {a_rad[i]:8.3f} {a_fwhm[i]:8.3f} '
                            f'{in_fratio[i]:6.2f} {out_fratio[i]:6.2f}\n')

        if args.summary is not None:
            nas = None if args.na is None else [args.na]
            if len(args.summary) == 0:
                summary_file = None
            else:
                summary_file = Path(args.summary).resolve()
                if summary_file.parent != oroot:
                    summary_file = oroot / summary_file.name
            plot.frd_plot([_ofile], nas=nas, ofile=summary_file)


