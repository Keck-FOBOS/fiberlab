"""
Script that simply converts images from their raw format to fits.
""" 

from . import scriptbase


class ConvertToFits(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        from pathlib import Path
        parser = super().get_parser(description='Convert images to fits files', width=width)
        parser.add_argument('root', default=str(Path().resolve()), type=str,
                            help='Directory with output files.')
        parser.add_argument('-i', '--image', default=None, type=str,
                            help='Name of single file to convert.  If provided, -s and -e '
                                 'arguments are ignored.')
        parser.add_argument('-s', '--search', default=None, type=str,
                            help='Search string for image names')
        parser.add_argument('-e', '--ext', default='.fit', type=str,
                            help='Image extension')
        parser.add_argument('-o', '--oroot', default=None, type=str,
                            help='Directory for output files.  Default is to write to the '
                                 'directory with the files to convert.')
        parser.add_argument('--overwrite', default=False, action='store_true',
                            help='Overwrite any existing files.')
        return parser

    @staticmethod
    def main(args):
        
        from pathlib import Path
        from IPython import embed
        from astropy.io import fits
        from ..io import bench_image

        # Set the root path
        root = Path(args.root).resolve()
        if not root.exists():
            raise FileNotFoundError(f'{root} is not a valid directory.')

        # Get the images to convert
        if args.image is None:
            search_str = f'*{args.ext}' if args.search is None else f'*{args.search}*{args.ext}'
            files = sorted(list(root.glob(search_str)))
        else:
            f = root / args.image
            if not f.exists():
                raise FileNotFoundError(f'{f} does not exist!')
            files = [f]

        # Get the output directory and make it if necessary
        oroot = root if args.oroot is None else Path(args.oroot).resolve()
        if not oroot.exists():
            oroot.mkdir(parents=True)

        # Read and convert the files
        for f in files:
            img = bench_image(f)
            ofile = oroot / f.with_suffix('.fits').name
            hdr = fits.Header()
            hdr['IFILE'] = str(f)
            fits.HDUList([fits.PrimaryHDU(header=hdr, data=img)
                          ]).writeto(ofile, overwrite=args.overwrite)
            print(f'Wrote {ofile}')


