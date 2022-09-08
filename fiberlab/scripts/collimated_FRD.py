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
        parser.add_argument('-o', '--oroot', default=str(Path().resolve()), type=str,
                            help='Directory for output files')
        parser.add_argument('-f', '--files', default=None, type=str,
                            help='Name of a file that provides 2 or 3 columns: (1) the '
                                 'files to analyze, (2) the background image to use for each '
                                 'file, and (3) the threshold to use for each file.  If this '
                                 'file is provided, any threshold is ignored and the root '
                                 'directory is not trolled for all bg*, z*, and a* files.  The '
                                 'last column with the thresholds can be omitted, which means '
                                 'the code will use the value provided on the command line (or '
                                 'its default).')
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

        # Analyze the two baseline images
        plot_file = oroot / f'{z0.with_suffix("").name}_qa.png'
        print(f'Analyzing {z0.name}')
        z0_rad, z0_peak, z0_fwhm \
                = collimated.collimated_farfield_output(z0, bkg_file=z0_bg, threshold=z0_thresh,
                                                        pixelsize=args.pixelsize,
                                                        plot_file=plot_file)
        plot_file = oroot / f'{z1.with_suffix("").name}_qa.png'
        print(f'Analyzing {z1.name}')
        z1_rad, z1_peak, z1_fwhm \
                = collimated.collimated_farfield_output(z1, bkg_file=z1_bg, threshold=z1_thresh,
                                                        pixelsize=args.pixelsize,
                                                        plot_file=plot_file)

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

        distance = args.sep/(1-z0_rad/z1_rad)

        if len(afiles) > 0:
            nangle = len(afiles)
            a_rad = numpy.zeros(nangle, dtype=float)
            a_peak = numpy.zeros(nangle, dtype=float)
            a_fwhm = numpy.zeros(nangle, dtype=float)
            for i in range(nangle):
                print(f'Analyzing {afiles[i].name}')
                plot_file = oroot / f'{afiles[i].with_suffix("").name}_qa.png'
                a_rad[i], a_peak[i], a_fwhm[i] \
                        = collimated.collimated_farfield_output(afiles[i], bkg_file=a_bg[i],
                                                                threshold=a_thresh[i],
                                                                pixelsize=args.pixelsize,
                                                                distance=distance,
                                                                plot_file=plot_file)
        else:
            a_rad = a_peak = a_fwhm = None

        # TODO: Save sigma and level

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
            f.write(f'# Distance from fiber output to z1 image: {distance:.2f} mm\n#\n')
            if a_rad is None:
                f.write(f'# No angle-sweep images to analyze\n')
            else:
                f.write('# Angle sweep results\n')
                f.write('# Radius and FWHM are in degrees.\n#\n')
                f.write(f'# {"FILE":>15} {"BKG":>15} {"THRESH":>6} {"PEAK":>10} {"RAD":>8} '
                        f'{"FWHM":>8}\n')
                for i in range(nangle):
                    f.write(f'  {afiles[i].name:>15} {a_bg[i].name:>15} {a_thresh[i]:6.1f} '
                            f'{a_peak[i]:10.2e} {a_rad[i]:8.3f} {a_fwhm[i]:8.3f}\n')


