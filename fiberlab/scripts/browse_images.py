"""
Script that simply browses through images in a directory.
""" 

from . import scriptbase


class BrowseImages(scriptbase.ScriptBase):

#    @classmethod
#    def name(cls):
#        return 'fiberlab_browse_images'

    @classmethod
    def get_parser(cls, width=None):
        from pathlib import Path
        parser = super().get_parser(description='Browse images in a directory', width=width)
        parser.add_argument('root', default=str(Path().resolve()), type=str,
                            help='Directory with output files.')
        parser.add_argument('-i', '--image', default=None, type=str,
                            help='Name of single file to show.  If provided, -s and -e arguments '
                                 'are ignored.')
        parser.add_argument('-s', '--search', default=None, type=str,
                            help='Search string for image names')
        parser.add_argument('-e', '--ext', default='.fit', type=str,
                            help='Image extension')
        parser.add_argument('-z', '--zlim', nargs=2, default=None, type=float,
                            help='The upper and lower values to use for the image plot limits.  '
                                 'Default is to plot +/- 5 sigma around the image mean.')
        return parser

    @staticmethod
    def main(args):

        from platform import system
        if system() == 'Windows':
            import matplotlib
            matplotlib.use('qtagg')

        from pathlib import Path
        from IPython import embed
        import numpy
        from matplotlib import pyplot
        from ..io import bench_image

        # Set the root path
        root = Path(args.root).resolve()
        if not root.exists():
            raise FileNotFoundError(f'{root} is not a valid directory.')

        if args.image is None:
            search_str = f'*{args.ext}' if args.search is None else f'*{args.search}*{args.ext}'
            files = sorted(list(root.glob(search_str)))
        else:
            f = root / args.image
            if not f.exists():
                raise FileNotFoundError(f'{f} does not exist!')
            files = [f]
        for f in files:
            img = bench_image(f)
            if args.zlim is None:
                mean = numpy.mean(img)
                std = numpy.std(img)
                zlim = [mean - 5*std, mean + 5*std]
            else:
                zlim = args.zlim

            base_width = 0.6
            aspect = img.shape[0]/img.shape[1]
            if aspect > 1:
                dx = base_width / aspect
                dy = base_width
            else:
                dx = base_width
                dy = base_width * aspect

            w,h = pyplot.figaspect(1)
            fig = pyplot.figure(figsize=(1.5*w,1.5*h))

            ax = fig.add_axes([0.5-dx/2, 0.15, dx, dy])
            implt = ax.imshow(img, origin='lower', interpolation='nearest',
                              vmin=zlim[0], vmax=zlim[1])
            cax = fig.add_axes([0.5-dx/2 + dx/5, 0.05, 3*dx/5, 0.01])
            fig.colorbar(implt, cax=cax, orientation='horizontal')

            ax = fig.add_axes([0.2, 0.15 + dy + 0.05, 0.6, 0.1])
            ax.hist(img.ravel(), bins=100, histtype='stepfilled', color='k', alpha=0.5, lw=0)
            ax.set_yscale('log')
            ax.set_title(f'{f.parent.name}/{f.name}')

            pyplot.show()
            fig.clear()
            pyplot.close(fig)


