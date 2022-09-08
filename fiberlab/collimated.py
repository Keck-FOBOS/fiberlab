"""
Module with methods used to analyze and plot collimated test images.
""" 

from IPython import embed

import numpy
from scipy import interpolate
from matplotlib import pyplot, ticker

from . import contour
from .io import bench_image

def default_threshold():
    """
    The default threshold to use for defining the contour used to determine the
    center of the output beam.
    """
    return 1.5

def collimated_farfield_output(img_file, bkg_file=None, threshold=None, pixelsize=None,
                               distance=None, plot_file=None, snr_img=False):
    """
    Analyze an output far-field image for a collimated input beam.

    Args:
        img_file (:obj:`str`, `Path`_):
            Image file
        bkg_file (:obj:`str`, `Path`_, optional):
            Background image to subtract
        threshold (:obj:`float`, optional):
            S/N threshold of the contour used to define the center of the ring.
            If None, see :func:`default_threshold`.
        pixelsize (:obj:`float`, optional):
            Size of the image pixels in mm.
        distance (:obj:`float`, optional):
            Distance from the fiber output to the detector in mm.
        plot_file (:obj:`str`, `Path`_, optional_):
            Name of the QA file to produce.  If ``'show'``, no file is produced,
            but the QA plot is displayed in a matplotlib window.
        snr_img (:obj:`bool`, optional):
            If creating the QA plot, plot the estimated S/N (used to set the
            contour level) instead of the measured counts.
    
    Returns:
        :obj:`tuple`: Three floating-point objects providing the radius at which
        the peak flux is found, the flux at the peak, and the width of the ring.
    """
    # Read in the image and do the basic background subtraction
    img = bench_image(img_file)
    if bkg_file is None:
        img_nobg = img.copy()
    else:
        bkg = bench_image(bkg_file)
        if img.shape != bkg.shape:
            raise ValueError('Image shape mismatch.')
        img_nobg = img - bkg

    # Get the contour to use for finding the image center
    _threshold = default_threshold() if threshold is None else threshold
    level, trace, trace_sig, trace_bkg = contour.get_contour(img_nobg, threshold=_threshold)
    # Remove any residual background
    img_nobg -= trace_bkg

    # Fit a Circle to the contour and get its center coordinates
    circ_p = contour.Circle(trace[:,0], trace[:,1]).fit()

    # Use the coordinates to compute the radius of each pixel
    ny, nx = img_nobg.shape
    x, y = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))
    circ_r, circ_theta = contour.Circle.polar(x, y, circ_p)
    r_units = 'pix'

    # Convert the radius from pixels to mm
    if pixelsize is not None :
        circ_r *= pixelsize 
        r_units = 'mm'
    # Convert the radius from mm to angle
    if pixelsize is not None and distance is not None :
        circ_r = numpy.degrees(numpy.arctan(circ_r/distance))
        r_units = 'deg'

    # Collapse the flux in radius
    srt = numpy.argsort(circ_r.ravel())
    radius = circ_r.ravel()[srt]
    flux = img_nobg.ravel()[srt]

    # Smooth it
    smooth_flux = contour.iterative_filter(flux, 301, 2) #, clip_iter=10, sigma=5.)

    # Find the radius of the peak
    peak_indx = numpy.argmax(smooth_flux)

    # Find the FWHM
    halfmax = smooth_flux[peak_indx]/2
    try:
        left = interpolate.interp1d(smooth_flux[:peak_indx], radius[:peak_indx])(halfmax)
    except:
        left = 0.
    right = interpolate.interp1d(smooth_flux[peak_indx:], radius[peak_indx:])(halfmax)

    # Generate a "model image"
    model = interpolate.interp1d(radius, smooth_flux)(circ_r)

    if plot_file is not None:
        collimated_farfield_output_plot(img_file, img_nobg, model, _threshold, level, trace,
                                        circ_p[0], circ_p[1], radius, flux, smooth_flux, peak_indx,
                                        left, right, snr_img=snr_img, r_units=r_units,
                                        ofile=None if plot_file == 'show' else plot_file)

    return radius[peak_indx], flux[peak_indx], right-left


def collimated_farfield_output_plot(img_file, img, model, threshold, level, trace, xc, yc, radius,
                                    flux, smooth_flux, peak_indx, left, right, snr_img=False,
                                    r_units='pix', ofile=None):
    """
    Diagnostic plot for the measurements of a collimated far-field output beam.

    Args:
        img_file (:obj:`str`, `Path`_):
            Image file
        img (`numpy.ndarray`_):
            The background subtracted far-field image data.
        model (`numpy.ndarray`_):
            The model of the far-field image.
        threshold (:obj:`float`):
            S/N threshold of the contour used to define the center of the ring.
        level (:obj:`float`):
            The level in the image that corresponds to the S/N threshold.
        trace (`numpy.ndarray`_):
            The contour tracing the outside of the ring used to define the ring
            center.
        xc (:obj:`float`):
            The best-fitting center of the ring in pixel coordinates.  Note this
            is along the *2nd* axis of ``img``.
        yc (:obj:`float`):
            The best-fitting center of the ring in pixel coordinates.  Note this
            is along the *1st* axis of ``img``.
        radius (`numpy.ndarray`_):
            A (sorted) 1D array with the radii of all pixels in ``img`` relative
            to the ring center.
        flux (`numpy.ndarray`_):
            A 1D array with the flux of all pixels in ``img`` sorted by radius
            relative to the ring center.
        smooth_flux (`numpy.ndarray`_):
            The smoothed version of the ``flux`` vector, used to measure the
            radius of the ring, its peak flux, and its full-width at half
            maximum.
        peak_indx (:obj:`int`):
            The index of the element in ``smooth_flux`` at which the peak (ring
            center) was found.
        left (:obj:`float`):
            The inner radius of the ring at half its maximum.
        right (:obj:`float`):
            The outer radius of the ring at half its maximum.
        snr_img (:obj:`bool`, optional):
            Plot the estimated S/N (used to set the contour level) instead of
            the measured counts.
        r_umits (:obj:`str`, optional):
            The units of the radius vector.
        ofile (:obj:`str`, `Path`_, optional_):
            Name of the QA file to produce.  If ``'show'``, no file is produced,
            but the QA plot is displayed in a matplotlib window.
    """
    ny, nx = img.shape
    aspect = ny/nx

    resid = img - model
    sig = level / threshold

    image_lim = contour.growth_lim(numpy.append(img, model), 0.99, fac=1.2)
    snr_lim = contour.growth_lim(img/sig, 0.99, fac=1.2)
    resid_lim = contour.growth_lim(resid, 0.90, fac=1.0, midpoint=0.)
    radius_lim = [0, 2*radius[peak_indx]]
    if radius_lim[1] < right:
        radius_lim[1] = 2*right

    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(1.5*w,1.5*h))
    
    dx = 0.31
    dy = dx*aspect
    buff = 0.01
    sx = 0.02
    ey = 0.98
    cmap = 'viridis'

    # Observed data
    ax = fig.add_axes([sx, ey-dy, dx, dy])
    ax.tick_params(which='both', left=False, bottom=False)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    if snr_img:
        imgplt = ax.imshow(img/sig, origin='lower', interpolation='nearest', cmap=cmap,
                           vmin=snr_lim[0], vmax=snr_lim[1])
    else:
        imgplt = ax.imshow(img, origin='lower', interpolation='nearest', cmap=cmap,
                           vmin=image_lim[0], vmax=image_lim[1])
    ax.scatter(xc, yc, marker='+', color='w', lw=2, zorder=4)
    cax = fig.add_axes([sx + dx/5, ey-dy-0.02, 3*dx/5, 0.01]) 
    cb = fig.colorbar(imgplt, cax=cax, orientation='horizontal')

    ax.text(0.05, 0.9, 'SNR' if snr_img else 'Data', ha='left', va='center', color='w',
            transform=ax.transAxes, bbox=dict(facecolor='k', alpha=0.3, edgecolor='none'))

    # "Model"
    ax = fig.add_axes([sx+dx+buff, ey-dy, dx, dy])
    ax.tick_params(which='both', left=False, bottom=False)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    imgplt = ax.imshow(model, origin='lower', interpolation='nearest', cmap=cmap,
                       vmin=image_lim[0], vmax=image_lim[1])
    ax.scatter(xc, yc, marker='+', color='w', lw=2, zorder=4)
    ax.plot(trace[:,0], trace[:,1], color='w', lw=0.5)
    cax = fig.add_axes([sx + dx+buff + dx/5, ey-dy-0.02, 3*dx/5, 0.01]) 
    cb = fig.colorbar(imgplt, cax=cax, orientation='horizontal')

    ax.text(0.05, 0.9, 'Model', ha='left', va='center', color='w', transform=ax.transAxes,
            bbox=dict(facecolor='k', alpha=0.3, edgecolor='none'))

    # Residuals
    ax = fig.add_axes([sx+2*(dx+buff), ey-dy, dx, dy])
    ax.tick_params(which='both', left=False, bottom=False)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    imgplt = ax.imshow(resid, origin='lower', interpolation='nearest', cmap='RdBu_r',
                       vmin=resid_lim[0], vmax=resid_lim[1])
    cax = fig.add_axes([sx + 2*(dx+buff) + dx/5, ey-dy-0.02, 3*dx/5, 0.01]) 
    cb = fig.colorbar(imgplt, cax=cax, orientation='horizontal')

    ax.text(0.05, 0.9, 'Residual', ha='left', va='center', color='w', transform=ax.transAxes,
            bbox=dict(facecolor='k', alpha=0.3, edgecolor='none'))
 
    # Flux vs radius
    ax = fig.add_axes([0.07, 0.17, 0.90, 0.48])
    ax.set_xlim(radius_lim)
    ax.minorticks_on()
    ax.tick_params(which='major', length=8, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=4, direction='in', top=True, right=True)
    ax.grid(True, which='major', color='0.8', zorder=0, linestyle='-')

    ax.scatter(radius, flux, marker='.', s=10, lw=0, alpha=.5, color='k', zorder=3)
    ax.plot(radius, smooth_flux, color='C3', zorder=4)

    ax.axvline(left, color='C0', lw=2, alpha=0.6, zorder=5)
    ax.axvline(right, color='C0', lw=2, alpha=0.6, zorder=5)

    ax.axvline(radius[peak_indx], color='C1', lw=1, ls='--', zorder=6)

    ax.text(0.5, -0.08, f'Radius [{r_units}]', ha='center', va='center', transform=ax.transAxes)

    ax.text(-0.05, -0.12, f'Directory: {img_file.parent.name}', ha='left', va='center',
            transform=ax.transAxes)
    ax.text(-0.05, -0.17, f'File: {img_file.name}', ha='left', va='center',
            transform=ax.transAxes)
    ax.text(-0.05, -0.22, f'Ring radius: {radius[peak_indx]:.2f}', ha='left', va='center',
            transform=ax.transAxes)
    ax.text(-0.05, -0.27, f'Ring FWHM: {right-left:.2f}', ha='left', va='center',
            transform=ax.transAxes)

    if ofile is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(ofile, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)


