"""
Module with methods used to analyze and plot collimated test images.

.. include:: ../include/links.rst
""" 

from IPython import embed

import numpy
from scipy import interpolate, ndimage
from matplotlib import pyplot, ticker

from . import contour
from .io import bench_image
from . import util


def default_threshold():
    """
    The default threshold to use for defining the contour used to determine the
    center of the output beam.
    """
    return 1.5


def collimated_farfield_output(img_file, bkg_file=None, threshold=None, pixelsize=None,
                               distance=None, plot_file=None, snr_img=False, window=None,
                               box=None, gau=None, dr=None, savgol=(301,2)):
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
        plot_file (:obj:`str`, `Path`_, optional):
            Name of the QA file to produce.  If ``'show'``, no file is produced,
            but the QA plot is displayed in a matplotlib window.
        snr_img (:obj:`bool`, optional):
            If creating the QA plot, plot the estimated S/N (used to set the
            contour level) instead of the measured counts.
        window (:obj:`float`, optional):
            Limit the plotted image regions to this times the best-fitting peak
            of the ring flux distribution.  If None, the full image is shown.
        box (:obj:`int`, optional):
            Boxcar average the image by this number of pixels.  If None, full
            resolution of image is used.
        gau (:obj:`float`, optional):
            Gaussian smooth the image used to measure the contour that is fit to
            find the beam center.  This provides the sigma of the 2D smoothing
            Gaussian in *binned* pixels (see ``box``).  Note the smoothed image
            is *not* used to measure the ring peak or width, once the center of
            the beam is measured.
        dr (:obj:`float`, optional):
            Radial binning step.  Units depend on values given for ``pixelsize``
            and ``distance``.  If both are None, the value is in pixels.  If
            ``pixelsize`` is provided, units are mm.  If both are provided,
            units are in deg.  If None, data is not binned, and the smoothed
            profile alone is used to find the peak and width.
        savgol (:obj:`tuple`, optional):
            Two parameters used for the the Savitzky-Golay filter applied to the
            data to "model" the flux.  These two parameters are (1)
            ``window_length`` and (2) ``polyorder`` as implemented by the
            ``scipy.signal.savgol_filter`` function.  When binning the data
            (``dr`` is provided), the first number in this tuple should be
            significantly smaller.

    Returns:
        :obj:`tuple`: Three floating-point objects providing the radius at which
        the peak flux is found, the flux at the peak, and the width of the ring.
    """
    # Read in the image and do the basic background subtraction
    img = bench_image(img_file)
    if box is not None:
        img = util.boxcar_average(img, box)*box**2
    if bkg_file is None:
        img_nobg = img.copy()
    else:
        bkg = bench_image(bkg_file)
        if box is not None:
            bkg = util.boxcar_average(bkg, box)*box**2
        if img.shape != bkg.shape:
            raise ValueError('Image shape mismatch.')
        img_nobg = img - bkg
    _pixelsize = pixelsize
    if box is not None:
        _pixelsize *= box

    # Get the contour to use for finding the image center
    _threshold = default_threshold() if threshold is None else threshold
    _img_nobg = img_nobg if gau is None else ndimage.gaussian_filter(img_nobg, sigma=gau)
    level, trace, trace_sig, trace_bkg = contour.get_contour(_img_nobg, threshold=_threshold)
    # Remove any residual background
    img_nobg -= trace_bkg

    # Fit a Circle to the contour and get its center coordinates
    circ_p = contour.Circle(trace[:,0], trace[:,1]).fit()

    # Use the coordinates to compute the radius of each pixel
    ny, nx = img_nobg.shape
    x, y = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))
    circ_r, circ_theta = contour.Circle.polar(x, y, circ_p)

    r_units, circ_r = contour.convert_radius(circ_r, pixelsize=_pixelsize, distance=distance)

    # Collapse the flux in radius
    srt = numpy.argsort(circ_r.ravel())
    radius = circ_r.ravel()[srt]
    flux = img_nobg.ravel()[srt]

    if dr is None:
        bin_radius = radius
        bin_flux = flux
        smooth_flux = contour.iterative_filter(bin_flux, savgol[0], savgol[1])
    else:
        bini = (radius/dr).astype(int)    
        nbin = numpy.amax(bini) + 1
        bin_flux = numpy.zeros(nbin, dtype=float)
        bin_radius = dr*(numpy.arange(nbin) + 0.5)
        for i in range(nbin):
            indx = bini == i
            if not numpy.any(indx):
                continue
            bin_flux[i] = numpy.mean(flux[indx])
#            bin_radius[i] = numpy.mean(radius[indx])

        # Smooth it
        smooth_flux = contour.iterative_filter(bin_flux, savgol[0], savgol[1]) #, clip_iter=10, sigma=5.)

    # Find the radius of the peak
    peak_indx = numpy.argmax(smooth_flux)

    # Find the FWHM
    halfmax = smooth_flux[peak_indx]/2
    try:
        left = interpolate.interp1d(smooth_flux[:peak_indx], bin_radius[:peak_indx])(halfmax)
    except:
        left = 0.
    right = interpolate.interp1d(smooth_flux[peak_indx:], bin_radius[peak_indx:])(halfmax)

    # Generate a "model image"

    if plot_file is not None:
        model = interpolate.interp1d(bin_radius, smooth_flux, bounds_error=False,
                                     fill_value=(smooth_flux[0], smooth_flux[-1]))(circ_r)
        collimated_farfield_output_plot(img_file, img_nobg, model, _threshold, level, trace,
                                        circ_p, bin_radius, bin_flux, smooth_flux, peak_indx,
                                        left, right, snr_img=snr_img, r_units=r_units,
                                        window=window, pixelsize=_pixelsize, distance=distance,
                                        ofile=None if plot_file == 'show' else plot_file)

    return bin_radius[peak_indx], bin_flux[peak_indx], right-left


def collimated_farfield_output_plot(img_file, img, model, threshold, level, trace, circ_p, radius,
                                    flux, smooth_flux, peak_indx, left, right, snr_img=False,
                                    r_units='pix', window=None, pixelsize=None, distance=None,
                                    ofile=None):
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
        circ_p (:obj:`tuple`):
            Tuple with the best-fitting parameters for the ring contour: x
            center (along 2nd axis), y center (along 1st axis), and radius.
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
        window (:obj:`float`, optional):
            Limit the plotted image regions to this times the best-fitting
            contour radius.  If None, the full image is shown.
        ofile (:obj:`str`, `Path`_, optional):
            Name of the QA file to produce.  If ``'show'``, no file is produced,
            but the QA plot is displayed in a matplotlib window.
    """
    xc, yc, rc = circ_p
    ny, nx = img.shape
    extent = [-0.5, nx-0.5, -0.5, ny-0.5]

    if window is None:
        aspect = ny/nx
        xlim = None
        ylim = None
    else:
        peak_r = contour.convert_radius(numpy.array([right]), pixelsize=pixelsize,
                                        distance=distance, inverse=True)[1][0] 
        xlim = [xc - window*peak_r, xc + window*peak_r]
        ylim = [yc - window*peak_r, yc + window*peak_r]
        aspect = 1.

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
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if snr_img:
        imgplt = ax.imshow(img/sig, origin='lower', interpolation='nearest', cmap=cmap,
                           vmin=snr_lim[0], vmax=snr_lim[1], extent=extent)
    else:
        imgplt = ax.imshow(img, origin='lower', interpolation='nearest', cmap=cmap,
                           vmin=image_lim[0], vmax=image_lim[1], extent=extent)
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
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    imgplt = ax.imshow(model, origin='lower', interpolation='nearest', cmap=cmap,
                       vmin=image_lim[0], vmax=image_lim[1], extent=extent)
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
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    imgplt = ax.imshow(resid, origin='lower', interpolation='nearest', cmap='RdBu_r',
                       vmin=resid_lim[0], vmax=resid_lim[1], extent=extent)
    cax = fig.add_axes([sx + 2*(dx+buff) + dx/5, ey-dy-0.02, 3*dx/5, 0.01]) 
    cb = fig.colorbar(imgplt, cax=cax, orientation='horizontal')

    ax.text(0.05, 0.9, 'Residual', ha='left', va='center', color='w', transform=ax.transAxes,
            bbox=dict(facecolor='k', alpha=0.3, edgecolor='none'))
 
    # Flux vs radius
    ax = fig.add_axes([0.08, 0.17, 0.90, 0.42])
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


