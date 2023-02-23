
from pathlib import Path

from IPython import embed

import numpy
from matplotlib import pyplot, ticker, rc, patches
from matplotlib.widgets import AxesWidget, RangeSlider, Cursor

from skimage import feature
from skimage import io

from . import scriptbase



class ImageCirclePatch:
    def __init__(self, fig, ax, par, incr=0.5, **kwargs):
        self.fig = fig
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.ax = ax if isinstance(ax, list) else [ax]
        self.nax = len(self.ax)
        self.par = par
        self.incr = incr
        self.circle_visible = True

        self.patch = [patches.Circle(par[:2], radius=par[2], **kwargs) for i in range(self.nax)]
        for i in range(self.nax):
            self.ax[i].add_patch(self.patch[i])

    def update_patches(self):
        for i in range(self.nax):
            self.patch[i].set(center=self.par[:2], radius=self.par[2])
            self.patch[i].set_visible(self.circle_visible)
        pyplot.draw()

    def on_press(self, event):
        if event.key == 'h':
            self.par[0] -= self.incr
        elif event.key == 'l':
            self.par[0] += self.incr
        elif event.key == 'k':
            self.par[1] += self.incr
        elif event.key == 'j':
            self.par[1] -= self.incr
        elif event.key == '+':
            self.par[2] += self.incr
        elif event.key == '-':
            self.par[2] -= self.incr
        elif event.key == 'I':
            self.incr *= 1.1
        elif event.key == 'i':
            self.incr /= 1.1
        elif event.key == 'p':
            print(f'{self.par[0]:8.2f} {self.par[1]:8.2f} {self.par[2]:8.2f} {self.incr:5.2f}')
        elif event.key == 'b':
            self.circle_visible = not self.circle_visible
        else:
            print(f'Unrecognized key: {event.key}')
        self.update_patches()

class UpdateableRangeSlider(RangeSlider):
    def __init__(self, ax, label, valmin, valmax, valinit=None, valfmt=None, closedmin=True,
                 closedmax=True, dragging=True, valstep=None, orientation="horizontal",
                 #track_color='lightgrey', handle_style=None,
                 **kwargs):

        super().__init__(ax, label, valmin, valmax, valinit=valinit, valfmt=valfmt,
                         closedmin=closedmin, closedmax=closedmax, dragging=dragging,
                         valstep=valstep, orientation=orientation, #track_color=track_color,
                         #handle_style=handle_style,
                         **kwargs)
        # Save the axes reference internally
        self.ax = ax
        self.label.set_position((0.5, 1.0))
        self.label.set_verticalalignment('bottom')
        self.label.set_horizontalalignment('center')
        self.label.set_weight('bold')

#    def update_range(self, rng): #, label):
#        self._active_handle = None
#        xy = self.poly.get_xy()
#        xy[:,0] = numpy.roll(numpy.repeat(rng, 3)[:-1],-1)
#        self.poly.set_xy(xy)
#        self.valmin, self.valmax = rng
#        self.valinit = numpy.array(rng)
#        self._handles[0].set_xdata(numpy.array([rng[0]]))
#        self._handles[1].set_xdata(numpy.array([rng[1]]))
#        self.ax.set_xlim(rng)
#        #self.label.set_text(label)
#        self.set_val(rng)


class UpdateableZRangeImage:
    def __init__(self, ax, sax, image_data, zmin, zmax):
        self.ax = ax

        self.data = image_data
        ny, nx = self.data.shape
        self.extent = [-0.5, nx-0.5, -0.5, ny-0.5]
        self.zmin = zmin
        self.zmax = zmax
        self.rng = (self.zmin, self.zmax)

        self.im = ax.imshow(self.data, interpolation='nearest', vmin=self.rng[0], vmax=self.rng[1],
                            extent=self.extent)

        # Colorbar range slider
        self.slider_ax = sax
        self.zslider = UpdateableRangeSlider(self.slider_ax, 'ZRange', self.rng[0], self.rng[1],
                                             valinit=self.rng)
        self.zslider.on_changed(self.change_range)

        pyplot.draw()

    def change_range(self, val):
        self.rng = (val[0], val[1])
        self.update_image()

    def update_image(self):
        self.im.set_clim(*self.rng)
#        self.zslider.update_range(self.rng)
        pyplot.draw()


class UpdateableCannyImage:
    def __init__(self, ax, sax, image_data, sigma=1., lo=0.1, hi=0.2):
        self.ax = ax

        self.data = image_data
        ny, nx = self.data.shape
        self.extent = [-0.5, nx-0.5, -0.5, ny-0.5]
        self.lo = lo
        self.hi = hi
        self.sigma = sigma
        self.rng = (self.lo, self.hi)

        self.canny = feature.canny(self.data, sigma=self.sigma, use_quantiles=True,
                                   low_threshold=self.lo, high_threshold=self.hi)
        self.im = ax.imshow(self.canny, interpolation='nearest', extent=self.extent)

        # Colorbar range slider
        self.slider_ax = sax
        self.zslider = UpdateableRangeSlider(self.slider_ax, 'Thresholds', 0.0, 1.0,
                                             valinit=self.rng)
        self.zslider.on_changed(self.change_range)

        pyplot.draw()

    def change_range(self, val):
        self.lo, self.hi = val
        self.rng = (self.lo, self.hi)
        self.update_image()

    def update_image(self):
        self.canny = feature.canny(self.data, sigma=self.sigma, use_quantiles=True,
                                   low_threshold=self.lo, high_threshold=self.hi)
        self.im.set_data(self.canny)
#        self.zslider.update_range(self.rng)
        pyplot.draw()

def mark_circles(ifile):

    _ifile = Path(ifile).resolve()
    if not _ifile.exists():
        raise FileNotFoundError(f'{_ifile} does not exist!')

    img = io.imread(_ifile).astype(float)
    if img.ndim > 2:
        img = numpy.mean(img, axis=-1)
    img -= numpy.mean(img)

    for key in pyplot.rcParams.keys():
        if 'keymap' not in key:
            continue
        pyplot.rcParams[key] = [] #None

    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(2.55*w,1.5*h))

    dx = 0.485
    ny, nx = img.shape
    aspect = ny/nx
    dy = dx*aspect*2
    buff = 0.01
    sx = 0.01
    ey = 0.99
    cmap = 'viridis'

    img_ax = fig.add_axes([sx, ey-dy, dx, dy])
    img_ax.tick_params(which='both', left=False, bottom=False)
    img_ax.xaxis.set_major_formatter(ticker.NullFormatter())
    img_ax.yaxis.set_major_formatter(ticker.NullFormatter())

    img_sax = fig.add_axes([sx + dx/10, ey-dy-0.1, 3*dx/5, 0.02]) 

    can_ax = fig.add_axes([sx+dx+buff, ey-dy, dx, dy], sharex=img_ax, sharey=img_ax)
    can_ax.tick_params(which='both', left=False, bottom=False)
    can_ax.xaxis.set_major_formatter(ticker.NullFormatter())
    can_ax.yaxis.set_major_formatter(ticker.NullFormatter())

    can_sax = fig.add_axes([sx+dx+buff + dx/10, ey-dy-0.1, 3*dx/5, 0.02]) 

    fiber_img = UpdateableZRangeImage(img_ax, img_sax, img, numpy.amin(img), numpy.amax(img))
    canny_img = UpdateableCannyImage(can_ax, can_sax, img, lo=0.8, hi=0.97)

    circle = ImageCirclePatch(fig, [img_ax, can_ax], [nx/2,ny/2,100], incr=0.5, 
                              facecolor='None', edgecolor='C3', lw=2, alpha=0.5)

#    imgplt = img_ax.imshow(img, interpolation='nearest', cmap=cmap, extent=extent)
#
#    imgplt = can_ax.imshow(canny_image, interpolation='nearest', cmap=cmap, extent=extent)

    pyplot.show()

    pyplot.rcdefaults()


class MarkCircles(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        parser = super().get_parser(description='Mark circles in an image', width=width)
        parser.add_argument('img_file', default=None, type=str,
                            help='File with the image to mark')
        return parser

    @staticmethod
    def main(args):

        mark_circles(args.img_file)


