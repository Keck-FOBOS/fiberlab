"""
Script that produces results for a collimated FRD test.
""" 

from pathlib import Path
import warnings

from IPython import embed

import numpy
from matplotlib import pyplot, rc, ticker, widgets
from matplotlib.backend_bases import MouseButton

from .. import fullcone
from .. import contour
from .. import util
from ..io import bench_image
from . import scriptbase


#-----------------------------------------------------------------------
# Pointer class
class Pointer(widgets.AxesWidget):
    """
    A pointer widget

    Args:
        ax (matplotlib.image.AxesImage):
            Returned object after running ``matplotlib.pyplot.imshow`` plotting
            an image to point within.
        kwargs (dict):
            Passed directly to ``matplotlib.widgets.Cursor``.
    """
    def __init__(self, ax, **kwargs):
        super().__init__(ax)

        if 'name' in kwargs:
            self.name = kwargs['name']
            kwargs.pop('name')
        else:
            self.name = None

        self.cursor = widgets.Cursor(ax, useblit=True, **kwargs)

        self.connect_event('button_press_event', self._button_update)
        self.connect_event('button_release_event', self._button_update)
        self.connect_event('key_press_event', self._key_update)
        self.connect_event('key_release_event', self._key_update)
        self.observers = {}
        self.drag_active = False
        self.pos = None
        self.action = None

    def _event_update(self, event, event_type):
        """update the pointer position"""
        if self.ignore(event):
            return

        if event_type not in ['button', 'key']:
            raise ValueError(f'Event type must be button or key, not {event_type}.')

        if event.name == f'{event_type}_press_event' and event.inaxes is self.ax:
            self.drag_active = True
            event.canvas.grab_mouse(self.ax)

        if not self.drag_active:
            return

        if event.name == f'{event_type}_release_event' \
                or (event.name == f'{event_type}_press_event' and event.inaxes is not self.ax):
            self.drag_active = False
            event.canvas.release_mouse(self.ax)
            return

        self._set_event(getattr(event, event_type), event.xdata, event.ydata)

    def _button_update(self, event):
        self._event_update(event, 'button')

    def _key_update(self, event):
        self._event_update(event, 'key')

    def _set_event(self, action, x, y):
        self.action = action
        self.pos = (x, y)
        if not self.eventson:
            return
        if self.action not in self.observers:
            print(f'No action: {action} ({self.name}, {x}, {y})')
            return
        if self.action in self.observers:
            self.observers[self.action](self.pos)

    def register(self, action, func):
        """
        Register a function to associate with a specific button or key press.
        """
        self.observers[action] = func

    def disconnect(self, action):
        """remove the observer with connection id *cid*"""
        try:
            del self.observers[action]
        except KeyError:
            pass


class UpdateableRangeSlider(widgets.RangeSlider):
    def __init__(self, ax, label, valmin, valmax, valinit=None, valfmt=None, closedmin=True,
                 closedmax=True, dragging=True, valstep=None, orientation="horizontal",
                 track_color='lightgrey', handle_style=None, **kwargs):

        super().__init__(ax, label, valmin, valmax, valinit=valinit, valfmt=valfmt,
                         closedmin=closedmin, closedmax=closedmax, dragging=dragging,
                         valstep=valstep, orientation=orientation, track_color=track_color,
                         handle_style=handle_style, **kwargs)
        self.label.set_position((0.5, 0.8))
        self.label.set_verticalalignment('bottom')
        self.label.set_horizontalalignment('center')
        self.label.set_weight('bold')
        # "Removes" the labels showing the current range
        self.valtext.set_visible(False)

    def update_range(self, rng, label=None):
        self._active_handle = None
        xy = self.poly.get_xy()
        xy[:,0] = numpy.roll(numpy.repeat(rng, 3)[:-1],-1)
        self.poly.set_xy(xy)
        self.valmin, self.valmax = rng
        self.valinit = numpy.array(rng)
        self._handles[0].set_xdata(numpy.array([rng[0]]))
        self._handles[1].set_xdata(numpy.array([rng[1]]))
        self.ax.set_xlim(rng)
        if label is not None:
            self.label.set_text(label)
        self.set_val(rng)


class UpdateableMaps:
    def __init__(self, image_plots, slider):
        self.image_plots = image_plots if isinstance(image_plots, list) else [image_plots]
        self.slider = slider
        self.slider.on_changed(self.change_range)

    def change_range(self, val):
        for imgplt in self.image_plots:
            imgplt.set_clim(*val)
        pyplot.draw()

    def change_maps(self, images):
        if len(images) != len(self.image_plots):
            raise ValueError(f'Mismatch between the number of images.  Got {len(images)}, '
                             f'expected {len(self.image_plots)}.')
        for plt, img in zip(self.image_plots, images):
            plt.set_data(img)
        rng = (numpy.amin([numpy.amin(img) for img in images]),
                numpy.amax([numpy.amax(img) for img in images]))
        self.slider.update_range(rng)
        self.change_range(rng)


class FarFieldPointer:
    """
    Pointer used when analyzing a far-field image.

    Args:
        s (matplotlib.collections.PathCollection):
            Result of an initial call to ``matplotlib.pyplot.scatter`` that
            plots a point on the image.
    """
    def __init__(self, img_plt, img_cen, img_contour, mod_plt, mod_cen, mod_contour, img_slider_ax,
                 res_plt, res_slider_ax, flux_sc, mod_flux_plt, circ_line, bkg_lo_line, bkg_hi_line,
                 ee_plt, text, ee, level, bkg_lim, pixelsize, distance):

        self.img_plt = img_plt
        self.img_cen = img_cen
        self.img_contour = img_contour

        self.mod_plt = mod_plt
        self.mod_cen = mod_cen
        self.mod_contour = mod_contour

        self.img_slider_ax = img_slider_ax

        self.res_plt = res_plt

        self.res_slider_ax = res_slider_ax

        self.flux_sc = flux_sc
        self.mod_flux_plt = mod_flux_plt
        self.circ_line = circ_line
        self.bkg_lo_line = bkg_lo_line
        self.bkg_hi_line = bkg_hi_line
        self.ee_plt = ee_plt

        self.text = text

        self.ee = ee
        self.level = level
        self.bkg_lim = bkg_lim
        self.pixelsize = pixelsize
        self.distance = distance

        # Value sliders
        #   - Images
        img_lim = self.img_plt.get_clim()
        self.img_slider = UpdateableRangeSlider(self.img_slider_ax, 'Data Range',
                                                img_lim[0], img_lim[1], valinit=img_lim)
        self.img_updater = UpdateableMaps([self.img_plt, self.mod_plt], self.img_slider)
        #   - Residuals
        res_lim = self.res_plt.get_clim()
        self.res_slider = UpdateableRangeSlider(self.res_slider_ax, 'Residual Range',
                                                res_lim[0], res_lim[1], valinit=res_lim)
        self.res_updater = UpdateableMaps(self.res_plt, self.res_slider)

        # Build the image pointer
        self.img_pointer = Pointer(self.img_plt.axes, name='image', color='C1', lw=0.5)
        self.img_pointer.register(MouseButton.LEFT, self.set_level)
        self.img_pointer.register('b', self.set_bkg_lower_image)
        self.img_pointer.register('B', self.set_bkg_upper_image)
        self.img_pointer.register('D', self.rm_bkg)
        self.img_pointer.register('C', self.print_contour)

        # NOTE: Have to use ee_plt here because it is the "top" axis, overlaying
        # the flux axis.  This is okay because we only care about the radius
        # position.
#        self.flx_pointer = Pointer(self.flux_sc.axes, name='scatter')
        self.flx_pointer = Pointer(self.ee_plt.axes, name='scatter', color='C1', lw=0.5)
        self.flx_pointer.register('b', self.set_bkg_lower_scatter)
        self.flx_pointer.register('B', self.set_bkg_upper_scatter)
        self.flx_pointer.register('D', self.rm_bkg)

    def set_level(self, pos):
        self.level = self.ee.img[int(pos[1]), int(pos[0])]
        self.update()

    def get_image_radius(self, pos):
        return numpy.sqrt((self.ee.circ_p[0] - pos[0])**2 + (self.ee.circ_p[1]-pos[1])**2) \
                / self.ee.circ_p[2]

    def set_bkg_lower_image(self, pos):
        self.set_bkg_lower(self.get_image_radius(pos))

    def set_bkg_lower_scatter(self, pos):
        r = contour.convert_radius(pos[0], pixelsize=self.pixelsize, distance=self.distance,
                                   inverse=True)[1] / self.ee.circ_p[2]
        self.set_bkg_lower(r)

    def set_bkg_lower(self, r):
        if self.bkg_lim is None:
            self.bkg_lim = [r, None]
        else:
            self.bkg_lim[0] = r
        self.update()

    def set_bkg_upper_image(self, pos):
        self.set_bkg_upper(self.get_image_radius(pos))

    def set_bkg_upper_scatter(self, pos):
        r = contour.convert_radius(pos[0], pixelsize=self.pixelsize, distance=self.distance,
                                   inverse=True)[1] / self.ee.circ_p[2]
        self.set_bkg_upper(r)

    def set_bkg_upper(self, r):
        if self.bkg_lim is None:
            warnings.warn('Ignoring input.  Set the lower limit first!')
            return
        self.bkg_lim[1] = r
        self.update()

    def rm_bkg(self, pos):
        self.bkg_lim = None
        if self.bkg_lo_line is not None:
            self.bkg_lo_line.remove()
            self.bkg_lo_line = None
        if self.bkg_hi_line is not None:
            self.bkg_hi_line.remove()
            self.bkg_hi_line = None
        self.update()

    def print_contour(self, pos):
        filename = input('File for contour data: ')
        if len(filename) == 0:
            for t in self.ee.trace:
                print(f'{t[0]:7.2f} {t[1]:7.2f}')
            return
        numpy.savetxt(filename, self.ee.trace, fmt=' %7.2f %7.2f')

    def update(self):
        if self.bkg_lim is not None and self.bkg_lim[1] is not None \
                and self.bkg_lim[0] > self.bkg_lim[1]:
            warnings.warn('Lower background boundary at larger radius than upper boundary.  '
                          'Swapping order.')
            self.bkg_lim = [self.bkg_lim[1], self.bkg_lim[0]]
        _img = numpy.ma.MaskedArray(self.ee.img, mask=numpy.logical_not(self.ee.inp_gpm))
        ee, xc, yc, rc, radius, flux, model_radius, model_flux, normalized_ee, ee90, rms \
                = reset_ee(_img, self.level, self.bkg_lim, self.pixelsize, self.distance)
        
        self.ee = ee
        
        self.img_plt.set_data(self.ee.img)
        self.img_cen.set_offsets((xc,yc))
        self.img_contour.set_data((self.ee.trace[:,0], self.ee.trace[:,1]))

        self.mod_plt.set_data(self.ee.model_img)
        self.mod_cen.set_offsets((xc,yc))
        self.mod_contour.set_data((self.ee.trace[:,0], self.ee.trace[:,1]))

        self.res_plt.set_data(self.ee.img - self.ee.model_img)

        self.flux_sc.set_offsets(numpy.column_stack((radius, flux)))
        self.mod_flux_plt.set_data((model_radius, model_flux))

        self.circ_line.set_data(([rc, rc], [0,1]))
        if self.bkg_lim is not None:
            if self.bkg_lim[1] is None:
                lo = self.bkg_lim[0]*rc
                hi = None
            else:
                lo = self.bkg_lim[0]*rc
                hi = self.bkg_lim[1]*rc

            if self.bkg_lo_line is None:
                self.bkg_lo_line \
                        = self.flux_sc.axes.axvline(lo, color='0.5', lw=1, ls='--', zorder=6) 
            else:
                self.bkg_lo_line.set_data(([lo,lo],[0,1]))

            if hi is None and self.bkg_hi_line is not None:
                self.bkg_hi_line.remove()
                self.bkg_hi_line = None
            elif hi is not None:
                if self.bkg_hi_line is None:
                    self.bkg_hi_line \
                            = self.flux_sc.axes.axvline(hi, color='0.5', lw=1, ls='--', zorder=6)
                else:
                    self.bkg_hi_line.set_data(([hi,hi],[0,1]))
        self.ee_plt.set_data((radius, normalized_ee))

        self.text['bkg'].set_text(f'{self.ee.bkg:.4e}')
        self.text['ee90'].set_text(f'{ee90:.4e}')
        self.text['flux'].set_text(f'{self.ee.ee_norm:.4e}')
        self.text['rms_full'].set_text(f'{rms[0]:.4e}')
        self.text['rms_90'].set_text(f'{rms[1]:.4e}')

        pyplot.draw()


def image_limits(img, model):
    _tmp = numpy.append(img, model)
    resid = img - model
    return [numpy.ma.amin(_tmp), numpy.ma.amax(_tmp)], \
            [numpy.ma.amin(resid), numpy.ma.amax(resid)]


def farfield_inspector(img_file, bkg_file=None, pixelsize=None, distance=None, smooth=False,
                       box=1):
    """
    Interactively analyze a far-field image.

    Args:
        img_file (str):
            Name of the image
        bkg_file (str, optional):
            Image to use for the background
        pixelsize (float, optional):
            Size of the detector pixels in mm.  If None, assumed to be unknown
            and units are set to be pixels.
        distance (float, optional):
            Distance from the output beam focus and the detector in mm.  If
            None, units are either mm or pixels in the detector plane.  If
            provided, radii in the image are expressed in angles.
        smooth (bool, optional):
            Smooth the EE curve to avoid interpolation errors.
        box (int, optional):
            Bin the input input to reduce its size.
    """
    if distance is not None and pixelsize is None:
        warnings.warn('Distance is meaningless if the pixelsize in mm is not provided.')
        distance = None

    img_file = Path(img_file).resolve()
    if not img_file.exists():
        raise FileNotFoundError(f'{img_file} does not exist!')
    bkg_file = None if bkg_file is None else Path(bkg_file).resolve()
    if bkg_file is not None and not bkg_file.exists():
        raise FileNotFoundError(f'{bkg_file} does not exist!')

    # Read in the image and do the basic background subtraction
    img = bench_image(img_file)
    if box is not None:
        img = util.boxcar_average(img, box)*box**2
    if bkg_file is None:
        img = numpy.ma.MaskedArray(img)
    else:
        bkg = bench_image(bkg_file)
        if box is not None:
            bkg = util.boxcar_average(bkg, box)*box**2
        if img.shape != bkg.shape:
            raise ValueError('Image shape mismatch.')
        img = numpy.ma.MaskedArray(img - bkg)

#    bkg, x, y, gpm = contour.fit_bg(img, (6,6), fititer=2)
#    img -= bkg

    if pixelsize is None:
        r_units = 'pix'
        _pixelsize = 1.
    else:
        r_units = 'mm'
        _pixelsize = pixelsize

    if box is not None:
        _pixelsize *= box

    # Quick and dirty starting position
    bkg = 0.
    level = 0.2*numpy.ma.amax(img) + 0.8*numpy.ma.amin(img)
    bkg_lim = None

    #img_file is only used for the file name and directory
    level, bkg_lim = interactive_ee(img_file, img, bkg, level, bkg_lim, _pixelsize, r_units,
                                    distance)

    result = f'Call fiberlab_fullcone_farfield using: --level {level:.2f}'
    if bkg_lim is not None:
        result += f' --bkg_lim {bkg_lim[0]:.2f}'
        if bkg_lim[1] is not None:
            result += f' {bkg_lim[1]:.2f}'
    if box is not None:
        result += f' --box {box}'
    print(result)


def reset_ee(img, level, bkg_lim, pixelsize, distance):
    # Perform the initial EE analysis
    if isinstance(img, numpy.ma.MaskedArray):
        _img = img.data
        mask = numpy.ma.getmaskarray(img)
    else:
        _img = img
        mask = None
    ee = fullcone.EECurve(_img, mask=mask, level=level, bkg_lim=bkg_lim)
    xc, yc, rc = ee.circ_p
    rc *= pixelsize
    radius = contour.convert_radius(ee.radius, pixelsize=pixelsize, distance=distance)[1]
    model_radius = contour.convert_radius(ee.model_radius, pixelsize=pixelsize,
                                          distance=distance)[1]

    ee90 = ee.ee_interpolator([0.9])[0]
    resid = ee.img - ee.model_img
    indx = ee.circ_r < ee90
    rms = (numpy.sqrt(numpy.mean((resid)**2)), numpy.sqrt(numpy.mean((resid[indx])**2)))    
    ee90 = contour.convert_radius(ee90, pixelsize=pixelsize, distance=distance)[1]

    return ee, xc, yc, rc, radius, ee.flux, model_radius, ee.model_flux, ee.ee/ee.ee_norm, \
                ee90, rms


def interactive_ee(img_file, img, bkg, level, bkg_lim, pixelsize, r_units, distance):

    # Perform the initial EE analysis
    ee, xc, yc, rc, radius, flux, model_radius, model_flux, normalized_ee, ee90, rms \
            = reset_ee(img, level, bkg_lim, pixelsize, distance)
    
    # Make the initial plots
    # - Image limits
    img_lim, res_lim = image_limits(ee.img, ee.model_img)
    rad_lim = [0, numpy.amax(radius)]
    # - Image coordinates
    ny, nx = ee.img.shape
    extent = [-0.5, nx-0.5, -0.5, ny-0.5]
    aspect = ny/nx
    # - Layout for image, model, residual
    dx = 0.31
    dy = dx*aspect
    buff = 0.01
    sx = 0.02
    ey = 0.98
    cmap = 'viridis'

    # Figure
    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(1.5*w,1.5*h))

    # Observed data axis
    img_ax = fig.add_axes([sx, ey-dy, dx, dy])
    img_ax.tick_params(which='both', left=False, bottom=False)
    img_ax.xaxis.set_major_formatter(ticker.NullFormatter())
    img_ax.yaxis.set_major_formatter(ticker.NullFormatter())
    img_cax = fig.add_axes([sx + dx/10, ey-dy-0.02, 3*dx/5, 0.01]) 
    img_cax.text(1.05, 0.4, 'Data', ha='left', va='center', color='k', transform=img_cax.transAxes)

    # Image plot
    img_plt = img_ax.imshow(ee.img, origin='lower', interpolation='nearest', cmap=cmap,
                            vmin=img_lim[0], vmax=img_lim[1], extent=extent)
    img_cb = fig.colorbar(img_plt, cax=img_cax, orientation='horizontal')
    
    # Center marker
    img_cen = img_ax.scatter(xc, yc, marker='+', color='w', lw=2, zorder=4)
    # Contour
    img_contour = img_ax.plot(ee.trace[:,0], ee.trace[:,1], color='w', lw=0.5)[0]

    # "Model" axis
    mod_ax = fig.add_axes([sx+dx+buff, ey-dy, dx, dy])
    mod_ax.tick_params(which='both', left=False, bottom=False)
    mod_ax.xaxis.set_major_formatter(ticker.NullFormatter())
    mod_ax.yaxis.set_major_formatter(ticker.NullFormatter())
    mod_cax = fig.add_axes([sx + dx+buff + dx/10, ey-dy-0.02, 3*dx/5, 0.01]) 
    mod_cax.text(1.05, 0.4, 'Model', ha='left', va='center', color='k', transform=mod_cax.transAxes)

    # Model plot
    mod_plt = mod_ax.imshow(ee.model_img, origin='lower', interpolation='nearest', cmap=cmap,
                            vmin=img_lim[0], vmax=img_lim[1], extent=extent)
    mod_cb = fig.colorbar(mod_plt, cax=mod_cax, orientation='horizontal')
    
    # Center marker
    mod_cen = mod_ax.scatter(xc, yc, marker='+', color='w', lw=2, zorder=4)
    # Contour
    mod_contour = mod_ax.plot(ee.trace[:,0], ee.trace[:,1], color='w', lw=0.5)[0]

    # Residuals axis
    res_ax = fig.add_axes([sx+2*(dx+buff), ey-dy, dx, dy])
    res_ax.tick_params(which='both', left=False, bottom=False)
    res_ax.xaxis.set_major_formatter(ticker.NullFormatter())
    res_ax.yaxis.set_major_formatter(ticker.NullFormatter())
    res_cax = fig.add_axes([sx + 2*(dx+buff) + dx/10, ey-dy-0.02, 3*dx/5, 0.01]) 
    res_cax.text(1.05, 0.4, 'Resid', ha='left', va='center', color='k', transform=res_cax.transAxes)

    # Residual plot
    res_plt = res_ax.imshow(ee.img - ee.model_img, origin='lower', interpolation='nearest',
                            cmap='RdBu_r', vmin=res_lim[0], vmax=res_lim[1], extent=extent)
    res_cb = fig.colorbar(res_plt, cax=res_cax, orientation='horizontal')

    # Flux vs radius
    flux_ax = fig.add_axes([0.08, 0.05, 0.84, 0.42])
    flux_ax.set_xlim(rad_lim)
    flux_ax.minorticks_on()
    flux_ax.tick_params(which='major', length=8, direction='in', top=True, right=True)
    flux_ax.tick_params(which='minor', length=4, direction='in', top=True, right=True)
    flux_ax.grid(True, which='major', color='0.8', zorder=0, linestyle='-')

    flux_sc = flux_ax.scatter(radius, flux, marker='.', s=10, lw=0, alpha=0.2, color='k', zorder=3)
    mod_flux_plt = flux_ax.plot(model_radius, model_flux, color='C3', zorder=4)[0]

    circ_line = flux_ax.axvline(rc, color='C3', lw=1, ls='--', zorder=6)

    if bkg_lim is not None:
        bkg_lo_line = flux_ax.axvline(bkg_lim[0]*rc, color='0.5', lw=1, ls='--', zorder=6)
        if bkg_lim[1] is None:
            bkg_hi_line = None
        else:
            bkg_hi_line = flux_ax.axvline(bkg_lim[1]*rc, color='0.5', lw=1, ls='--', zorder=6)
    else:
        bkg_lo_line = None
        bkg_hi_line = None

    flux_ax.text(0.5, -0.08, f'Radius [{r_units}]', ha='center', va='center',
                 transform=flux_ax.transAxes)

    ee_ax = flux_ax.twinx()
    ee_ax.set_xlim(rad_lim)
    ee_ax.spines['right'].set_color('0.3')
    ee_ax.tick_params(which='both', axis='y', direction='in', colors='0.3')
    ee_ax.yaxis.label.set_color('0.3')
    ee_ax.set_ylabel('Enclosed Energy')
    ee_ax.axhline(1.0, color='0.5', lw=1, ls='--', zorder=6)

    ee_plt = ee_ax.plot(radius, normalized_ee, color='0.3', lw=0.5, zorder=5)[0]

    # Slider axes
    img_slider_ax = fig.add_axes([sx, ey-dy-0.11, 0.4, 0.04])
    res_slider_ax = fig.add_axes([sx, ey-dy-0.17, 0.4, 0.04])

    text = {}
    text['bkg'] = fig.text(0.92, 0.600, f'{ee.bkg:.4e}', ha='right', va='center')
    text['ee90'] = fig.text(0.92, 0.575, f'{ee90:.4e}', ha='right', va='center')
    text['flux'] = fig.text(0.92, 0.550, f'{ee.ee_norm:.4e}', ha='right', va='center')
    text['rms_full'] = fig.text(0.92, 0.525, f'{rms[0]:.4e}', ha='right', va='center')
    text['rms_90'] = fig.text(0.92, 0.500, f'{rms[1]:.4e}', ha='right', va='center')

    fig.text(0.7, 0.600, f'BKG', ha='left', va='center')
    fig.text(0.7, 0.575, f'EE90', ha='left', va='center')
    fig.text(0.7, 0.550, f'FLUX', ha='left', va='center')
    fig.text(0.7, 0.525, f'RMS', ha='left', va='center')
    fig.text(0.7, 0.500, f'RMS90', ha='left', va='center')

    # Pointer in data image
    pointer = FarFieldPointer(img_plt, img_cen, img_contour, mod_plt, mod_cen, mod_contour,
                              img_slider_ax, res_plt, res_slider_ax, flux_sc, mod_flux_plt,
                              circ_line, bkg_lo_line, bkg_hi_line, ee_plt, text, ee,
                              level, bkg_lim, pixelsize, distance)

    pyplot.show()

    return pointer.level, pointer.bkg_lim


class FarFieldInspector(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        from pathlib import Path
        parser = super().get_parser(description='Interactively inspect a far-field image',
                                    width=width)
        parser.add_argument('img_file', default=None, type=str,
                            help='File with far-field output image from a collimated input')
        parser.add_argument('--bkg_file', default=None, type=str,
                            help='File with only background flux')
        parser.add_argument('-p', '--pixelsize', default=None, type=float,
                            help='Size of the image camera pixels in mm.')
        parser.add_argument('-d', '--distance', default=None, type=float,
                            help='Distance between the fiber output and the camera detector')
        parser.add_argument('--smooth', default=False, action='store_true',
                            help='Smooth the EE curve to limit interpolation errors')
        parser.add_argument('--box', default=None, type=int,
                            help='Boxcar average the image before analyzing it')
        return parser

    @staticmethod
    def main(args):

        farfield_inspector(args.img_file, bkg_file=args.bkg_file, pixelsize=args.pixelsize,
                           distance=args.distance, smooth=args.smooth, box=args.box)




