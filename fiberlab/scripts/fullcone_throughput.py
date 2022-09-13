"""
Script that produces results for a collimated FRD test.
""" 

from . import scriptbase


class FullConeThroughput(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        from pathlib import Path
        parser = super().get_parser(description='Calculate results for a full-cone far-field test.',
                                    width=width)
        parser.add_argument('inp_img', default=None, type=str,
                            help='File with an image of the input beam')
        parser.add_argument('out_img', default=None, type=str,
                            help='File with an image of the output beam')
        parser.add_argument('-b', '--bkg_img', default=None, type=str,
                            help='File with only background flux')
        parser.add_argument('-t', '--threshold', default=1.5, type=float,
                            help='S/N threshold that sets the contour used to identify the center '
                                 'of the output ring.')
        return parser

    @staticmethod
    def main(args):

        import time
        from pathlib import Path
        import warnings

        from IPython import embed

        from .. import fullcone

        inp_img = Path(args.inp_img).resolve()
        if not inp_img.exists():
            raise FileNotFoundError(f'{inp_img} does not exist!')

        out_img = Path(args.out_img).resolve()
        if not out_img.exists():
            raise FileNotFoundError(f'{out_img} does not exist!')

        bkg_img = None if args.bkg_img is None else Path(args.bkg_img).resolve()
        if bkg_img is not None and not bkg_img.exists():
            raise FileNotFoundError(f'{bkg_img} does not exist!')

        # Analyze the image
        inp_flux, out_flux, throughput \
                = fullcone.fullcone_throughput(inp_img, out_img, bkg_file=bkg_img,
                                               threshold=args.threshold, local_bg_fac=2.,
                                               local_iter=4)

        print('# Result from fobos_fullcone_throughput script')
        print(f'# Written: {time.strftime("%a %d %b %Y %H:%M:%S",time.localtime())}')
        print(f'# Input image: {inp_img.name}')
        print(f'# Output image: {out_img.name}')
        if bkg_img is not None:
            print(f'# Background image: {bkg_img.name}')
        print(f'# Input flux: {inp_flux:10.3e}')
        print(f'# Output flux: {out_flux:10.3e}')
        print(f'# Throughput: {throughput:6.3f}')


