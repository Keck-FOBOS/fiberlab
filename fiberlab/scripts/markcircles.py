
from . import scriptbase


class MarkCircles(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        parser = super().get_parser(description='Mark circles in an image', width=width)
        parser.add_argument('img_file', default=None, type=str,
                            help='File with the image to mark')
        return parser

    @staticmethod
    def main(args):

        from pathlib import Path

        from IPython import embed

        import numpy
        from matplotlib import pyplot, ticker, rc, colors

        from skimage import feature
        from skimage import io

        ifile = Path(args.img_file).resolve()
        img = numpy.mean(io.imread(ifile).astype(float), axis=-1)
        img -= numpy.mean(img)

        canny_image = feature.canny(img, sigma=1.0, use_quantiles=True, low_threshold=0.8,
                                    high_threshold=0.95)

        ny, nx = img.shape
        extent = [-0.5, nx-0.5, -0.5, ny-0.5]
        aspect = ny/nx

        w,h = pyplot.figaspect(1)
        fig = pyplot.figure(figsize=(2*w,1*h))

        dx = 0.48
        dy = dx*aspect*2
        buff = 0.01
        sx = 0.02
        ey = 0.98
        cmap = 'viridis'

        img_ax = fig.add_axes([sx, ey-dy, dx, dy])
        img_ax.tick_params(which='both', left=False, bottom=False)
        img_ax.xaxis.set_major_formatter(ticker.NullFormatter())
        img_ax.yaxis.set_major_formatter(ticker.NullFormatter())
        imgplt = img_ax.imshow(img, interpolation='nearest', cmap=cmap, extent=extent)

    #    ax.scatter(xc, yc, marker='+', color='w', lw=2, zorder=4)
        cax = fig.add_axes([sx + dx/5, ey-dy-0.02, 3*dx/5, 0.01]) 
        cb = fig.colorbar(imgplt, cax=cax, orientation='horizontal')

        can_ax = fig.add_axes([sx+dx+buff, ey-dy, dx, dy], sharex=img_ax, sharey=img_ax)
        can_ax.tick_params(which='both', left=False, bottom=False)
        can_ax.xaxis.set_major_formatter(ticker.NullFormatter())
        can_ax.yaxis.set_major_formatter(ticker.NullFormatter())
        imgplt = can_ax.imshow(canny_image, interpolation='nearest', cmap=cmap, extent=extent)

        pyplot.show()




        #embed()
        exit()

