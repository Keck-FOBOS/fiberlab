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
        parser.add_argument('ofile', default='collimated_FRD.txt', type=str,
                            help='Output file for measurements')
        parser.add_argument('-p', '--pixelsize', default=0.018, type=float,
                            help='Size of the image camera pixels in mm.')
        parser.add_argument('-s', '--sep', default=3.82, type=float,
                            help='Known separation (in mm) between the "z" images used to '
                                 'calculate the distance from the fiber output to the main '
                                 'imaging position.')
        parser.add_argument('-t', '--threshold', default=default_threshold(), type=float,
                            help='S/N threshold that sets the contour used to identify the center '
                                 'of the output ring.')
        parser.add_argument('-r', '--ring_box', default=None, type=float,
                            help='Limit the plotted image regions to this times the best-fitting '
                                 'peak of the ring flux distribution.  If None, the full image '
                                 'is shown.')
        parser.add_argument('-o', '--oroot', default=str(Path().resolve()), type=str,
                            help='Directory for output files')
        parser.add_argument('-f', '--files', default=None, type=str,
                            help='Name of a file that provides 2 or 3 columns: (1) the '
                                 'files to analyze, (2) the background image to use for each '
                                 'file, and (3) the threshold to use for each file.  If this '
                                 'file is provided, any threshold is ignored and the root '
                                 'directory is not trolled for all bg* and z* files.  The '
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
        from scipy import interpolate

        from .. import fullcone
        from .. import io

        # Set the root path
        root = Path(args.root).resolve()
        if not root.exists():
            raise FileNotFoundError(f'{root} is not a valid directory.')

        oroot = Path(args.oroot).resolve()
        if not oroot.exists():
            oroot.mkdir(parents=True)

        # Get files
        z0, z0_bg, z0_thresh, z1, z1_bg, z1_thresh \
                = io.gather_fullcone_file_list(root, par=args.files, threshold=args.threshold)

        # Analyze the two baseline images
        plot_file = oroot / f'{z0.with_suffix("").name}_qa.png'
        print(f'Analyzing {z0.name}')
        z0_radius_img, z0_img, z0_radius, z0_flux \
                = fullcone.fullcone_farfield_output(z0, bkg_file=z0_bg, threshold=z0_thresh,
                                                    pixelsize=args.pixelsize, plot_file=plot_file,
                                                    ring_box=args.ring_box, local_iter=3, local_bg_fac=3.)
        plot_file = oroot / f'{z1.with_suffix("").name}_qa.png'
        print(f'Analyzing {z1.name}')
        z1_radius_img, z1_img, z1_radius, z1_flux \
                = fullcone.fullcone_farfield_output(z1, bkg_file=z1_bg, threshold=z1_thresh,
                                                    pixelsize=args.pixelsize, plot_file=plot_file,
                                                    ring_box=args.ring_box, local_iter=3, local_bg_fac=3.)

        # Growth 
        z0_growth = numpy.cumsum(z0_flux)
        z0_growth /= z0_growth[-1]
        z1_growth = numpy.cumsum(z1_flux)
        z1_growth /= z1_growth[-1]

        # Get the distance by sampling the radius at which each EE curve meets a
        # given growth fraction
        ee_sample = numpy.linspace(0.01, 0.99, 99)

        indx = (ee_sample >= 0.1) & (ee_sample <= 0.9)

        ee_r_z0 = interpolate.interp1d(z0_growth, z0_radius)(ee_sample)
        ee_r_z1 = interpolate.interp1d(z1_growth, z1_radius)(ee_sample)

        z0_distance = args.sep/(ee_r_z1/ee_r_z0-1)
        z1_distance = args.sep/(1-ee_r_z0/ee_r_z1)

        med_z0_distance = args.sep/(numpy.median(ee_r_z1[indx]/ee_r_z0[indx])-1)
        med_z1_distance = args.sep/(1-numpy.median(ee_r_z0[indx]/ee_r_z1[indx]))

        ee_fratio_1 = med_z1_distance / 2 / ee_r_z1
        ee_fratio_0 = med_z0_distance / 2 / ee_r_z0

        # Main output file
        _ofile = Path(args.ofile).resolve()
        if _ofile.parent != oroot:
            _ofile = oroot / _ofile.name

        header = 'Result from fobos_fullcone_ee_fratio script\n' \
                 f'Written: {time.strftime("%a %d %b %Y %H:%M:%S",time.localtime())}\n\n' \
                 f'Top-level directory: {root}\n\n' \
                 f'Pixelsize: {args.pixelsize} mm\n' \
                 f'Z0 Image:         {z0.name}\n' \
                 f'Z0 Background:    {z0_bg.name}\n' \
                 f'Z0 S/N Threshold: {z0_thresh:.1f}\n' \
                 f'Median Z0 distance (0.1 < EE < 0.9): {med_z0_distance:.2f} mm\n' \
                 f'Z1 Image:         {z1.name}\n' \
                 f'Z1 Background:    {z1_bg.name}\n' \
                 f'Z1 S/N Threshold: {z1_thresh:.1f}\n' \
                 f'Median Z1 distance (0.1 < EE < 0.9): {med_z1_distance:.2f} mm\n\n' \
                 'EE is the fractional inclosed energy\n' \
                 'R0 is the radius in mm at the detector plane at the closest (z0) image\n' \
                 'R1 is the radius in mm at the detector plane at the farthest (z1) image\n' \
                 'F0 is the distance in mm to the closest (z0) image using R0\n' \
                 'F1 is the distance in mm to the farthest (z1) image using R1\n' \
                 'f0 is the focal ratio assuming the median z0 distance\n' \
                 'f1 is the focal ratio assuming the median z1 distance\n\n' \
                 f'{"EE":>6} {"R0":>6} {"R1":>6} {"F0":>6} {"F1":>6} {"f0":>6} {"f1":>6}'

        numpy.savetxt(_ofile, numpy.column_stack((ee_sample, ee_r_z0, ee_r_z1, z0_distance,
                                                      z1_distance, ee_fratio_0, ee_fratio_1)),
                      fmt=['%8.2f', '%6.2f', '%6.2f', '%6.2f', '%6.2f', '%6.2f', '%6.2f'],
                      header=header)


