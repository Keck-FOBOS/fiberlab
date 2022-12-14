"""
Module with methods used to analyze and plot full-cone test images.

.. include:: ../include/links.rst
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
    return 3.0


def fullcone_farfield_output(img_file, bkg_file=None, threshold=None, clip_iter=10,
                             sigma_lower=100., sigma_upper=3., local_bg_fac=None, local_iter=1,
                             pixelsize=None, distance=None, plot_file=None, snr_img=False,
                             ring_box=None):
    """
    Analyze an output far-field image for a full-cone input beam.

    Args:
        img_file (:obj:`str`, `Path`_):
            Image file
        bkg_file (:obj:`str`, `Path`_, optional):
            Background image to subtract
        threshold (:obj:`float`, optional):
            S/N threshold of the contour used to define the center of the ring.
            If None, see :func:`default_threshold`.
        local_bg_fac (:obj:`float`, optional):
            Number of HWHM at which to determine the local background
            level.  If None, no local background is estimated.
        local_iter (:obj:`int`, optional):
            Number of iterations used for determining the local
            background.  Ignored if ``local_bg_fac`` is None.
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
        ring_box (:obj:`float`, optional):
            Limit the plotted image regions to this times the best-fitting peak
            of the ring flux distribution.  If None, the full image is shown.
    
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
    level, trace, trace_sig, trace_bkg \
            = contour.get_contour(img_nobg, threshold=_threshold, clip_iter=clip_iter,
                                  sigma_lower=sigma_lower, sigma_upper=sigma_upper)
    # Remove any residual background
    img_nobg -= trace_bkg

    # Fit a Circle to the contour and get its center coordinates
    circ_p = contour.Circle(trace[:,0], trace[:,1]).fit()

    # Use the coordinates to compute the radius of each pixel
    ny, nx = img_nobg.shape
    x, y = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))
    circ_r, circ_theta = contour.Circle.polar(x, y, circ_p)

    # Construct 1D vectors with data sorted by radius:
    srt = numpy.argsort(circ_r.ravel())
    radius = circ_r.ravel()[srt]
    #   - Flux
    flux = img_nobg.ravel()[srt]
    #   - Aperture area
    area = numpy.pi*radius**2
    #   - Cumulative sum of the flux to get the EE (growth) curve
    ee = numpy.cumsum(flux)

    # Iteratively determine a local background, if requested
    bg_r = None
    local_bg = 0.
    if local_bg_fac is not None:
        for i in range(local_iter):
            # Get the half-width of the EE curve
            hwhm = interpolate.interp1d(ee/ee[-1], radius)([0.5])[0]
            # Get the background in the selected regions
            bg_r = local_bg_fac*hwhm
            indx = radius > bg_r
            _local_bg = contour.get_bg(flux[indx], clip_iter=clip_iter, sigma_lower=sigma_lower,
                                       sigma_upper=sigma_upper)[0]
            # Subtract it from the image and flux
            img_nobg -= _local_bg
            flux -= _local_bg
            # Recompute the EE curve
            ee = numpy.cumsum(flux)
            # Add it to the total
            local_bg += _local_bg

    # Construct a model of the luminosity distribution using the EE
    # curve
    # - Sample the measured EE curve at discrete radii
    model_radius = numpy.linspace(0,max(radius),500)[1:]
    model_ee = numpy.zeros_like(model_radius)
    indx = (model_radius > numpy.amin(radius)) & (model_radius < numpy.amax(radius))
    model_ee[indx] = interpolate.interp1d(radius, ee)(model_radius[indx])
    # - Handle extrapolation
    indx = (model_radius <= numpy.amin(radius))
    if any(indx):
        model_ee[indx] = ee[0]
    indx = (model_radius >= numpy.amax(radius))
    if any(indx):
        model_ee[indx] = ee[-1]
    # - Construct the model flux as the derivative of the EE curve
    model_flux = numpy.append(model_ee[0]/model_radius[0]**2,
                              numpy.diff(model_ee)/numpy.diff(model_radius**2)) / numpy.pi
    # - Interpolate the 1D model into a 2D image
    model_img = numpy.zeros_like(img)
    indx = (circ_r > model_radius[0]) & (circ_r < model_radius[-1])
    model_img[indx] = interpolate.interp1d(model_radius, model_flux)(circ_r[indx])
    # - Handle extrapolation
    indx = circ_r <= model_radius[0]
    if numpy.any(indx):
        model_img[indx] = model_flux[0]
    indx = circ_r >= model_radius[-1]
    if numpy.any(indx):
        model_img[indx] = model_flux[-1]

    r_units, circ_r = contour.convert_radius(circ_r, pixelsize=pixelsize, distance=distance)
    radius = contour.convert_radius(radius, pixelsize=pixelsize, distance=distance)[1]
    model_radius = contour.convert_radius(model_radius, pixelsize=pixelsize, distance=distance)[1]


    if plot_file is not None:
        fullcone_farfield_output_plot(img_file, img_nobg, model_img, _threshold, level, trace,
                                      circ_p, radius, flux, model_radius, model_flux, model_img,
                                      snr_img=snr_img, r_units=r_units, ring_box=ring_box,
                                      pixelsize=pixelsize, distance=distance,
                                      ofile=None if plot_file == 'show' else plot_file)

    return circ_r, img_nobg, radius, flux


def fullcone_farfield_output_plot(img_file, img, model, threshold, level, trace, circ_p, radius,
                                  flux, model_radius, model_flux, model_img, snr_img=False,
                                  r_units='pix', ring_box=None, pixelsize=None, distance=None,
                                  ofile=None):
    """
    UPDATE
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
        ring_box (:obj:`float`, optional):
            Limit the plotted image regions to this times the best-fitting
            contour radius.  If None, the full image is shown.
        ofile (:obj:`str`, `Path`_, optional):
            Name of the QA file to produce.  If ``'show'``, no file is produced,
            but the QA plot is displayed in a matplotlib window.
    """

    growth = numpy.cumsum(flux)
    growth /= growth[-1]
    right = interpolate.interp1d(growth, radius)(0.9)

    xc, yc, rc = circ_p
    ny, nx = img.shape
    extent = [-0.5, nx-0.5, -0.5, ny-0.5]
    if ring_box is None:
        aspect = ny/nx
        xlim = None
        ylim = None
    else:
#        peak_r = contour.convert_radius(numpy.array([radius[peak_indx]]), pixelsize=pixelsize,
#                                        distance=distance, inverse=True)[1][0] 
        peak_r = contour.convert_radius(numpy.array([right]), pixelsize=pixelsize,
                                        distance=distance, inverse=True)[1][0] 
        xlim = [xc - ring_box*peak_r, xc + ring_box*peak_r]
        ylim = [yc - ring_box*peak_r, yc + ring_box*peak_r]
        aspect = 1.

    resid = img - model
    sig = level / threshold

    image_lim = contour.growth_lim(numpy.append(img, model), 0.99, fac=1.2)
    snr_lim = contour.growth_lim(img/sig, 0.99, fac=1.2)
    resid_lim = contour.growth_lim(resid, 0.90, fac=1.0, midpoint=0.)
    radius_lim = [0, 1.5*right]

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
    ax = fig.add_axes([0.08, 0.17, 0.84, 0.42])
    ax.set_xlim(radius_lim)
    ax.minorticks_on()
    ax.tick_params(which='major', length=8, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=4, direction='in', top=True, right=True)
    ax.grid(True, which='major', color='0.8', zorder=0, linestyle='-')

    ax.scatter(radius, flux, marker='.', s=10, lw=0, alpha=0.2, color='k', zorder=3)
    ax.plot(model_radius, model_flux, color='C3', zorder=4)

    ax.axvline(right, color='C1', lw=1, ls='--', zorder=6)

    ax.text(0.5, -0.08, f'Radius [{r_units}]', ha='center', va='center', transform=ax.transAxes)

    ax.text(-0.05, -0.12, f'Directory: {img_file.parent.name}', ha='left', va='center',
            transform=ax.transAxes)
    ax.text(-0.05, -0.17, f'File: {img_file.name}', ha='left', va='center',
            transform=ax.transAxes)
    ax.text(-0.05, -0.22, f'EE90 radius: {right:.2f}', ha='left', va='center',
            transform=ax.transAxes)
#    ax.text(-0.05, -0.27, f'Ring FWHM: {right-left:.2f}', ha='left', va='center',
#            transform=ax.transAxes)

    axt = ax.twinx()
    axt.set_xlim(radius_lim)
    axt.spines['right'].set_color('0.3')
    axt.tick_params(which='both', axis='y', direction='in', colors='0.3')
    axt.yaxis.label.set_color('0.3')
    axt.plot(radius, growth, color='0.3', lw=0.5, zorder=5)
    #axt.grid(True, which='major', color='C0', zorder=0, linestyle=':')
    axt.set_ylabel('Enclosed Energy')

    if ofile is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(ofile, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)


def fullcone_throughput(inp_img, out_img, bkg_file=None, threshold=None, clip_iter=10,
                        sigma_lower=100., sigma_upper=3., local_bg_fac=None, local_iter=1):
    """
    Analyze an output far-field image for a full-cone input beam.

    Args:
        inp_img (:obj:`str`, `Path`_):
            Image of the input beam
        out_img (:obj:`str`, `Path`_):
            Image of the output beam
        bkg_file (:obj:`str`, `Path`_, optional):
            Background image to subtract
        threshold (:obj:`float`, optional):
            S/N threshold of the contour used to define the center of the ring.
            If None, see :func:`default_threshold`.
        clip_iter (:obj:`int`, optional):
            Number of clipping iterations used to measure background
        sigma_lower (:obj:`float`, optional):
            Lower sigma rejection used to measure background
        sigma_upper (:obj:`float`, optional):
            Upper sigma rejection used to measure background
        local_bg_fac (:obj:`float`, optional):
            Number of HWHM at which to determine the local background
            level.  If None, no local background is estimated.
        local_iter (:obj:`int`, optional):
            Number of iterations used for determining the local
            background.  Ignored if ``local_bg_fac`` is None.
    """
    # Read in the image and do the basic background subtraction
    inp_data = bench_image(inp_img)
    out_data = bench_image(out_img)
    if bkg_file is not None:
        bkg = bench_image(bkg_file)
        if inp_data.shape != bkg.shape:
            raise ValueError('Image shape mismatch.')
        inp_data -= bkg
        out_data -= bkg

    inp_radius, inp_ee = ee_curve(inp_data, threshold=threshold, clip_iter=clip_iter,
                                  sigma_lower=sigma_lower, sigma_upper=sigma_upper,
                                  local_iter=local_iter, local_bg_fac=local_bg_fac)
    inp_hwhm = interpolate.interp1d(inp_ee/inp_ee[-1], inp_radius)([0.5])[0]
    inp_flux = numpy.mean(inp_ee[inp_radius > 2*inp_hwhm])

    out_radius, out_ee = ee_curve(out_data, threshold=threshold, clip_iter=clip_iter,
                                  sigma_lower=sigma_lower, sigma_upper=sigma_upper,
                                  local_iter=local_iter, local_bg_fac=local_bg_fac)

    out_hwhm = interpolate.interp1d(out_ee/out_ee[-1], out_radius)([0.5])[0]
    out_flux = numpy.mean(out_ee[out_radius > 2*out_hwhm])

    return inp_flux, out_flux, out_flux/inp_flux


def ee_curve(img, threshold=None, clip_iter=10, sigma_lower=100., sigma_upper=3.,
             local_bg_fac=None, local_iter=1):

    # Get the ee curve for the input image
    _threshold = default_threshold() if threshold is None else threshold
    level, trace, trace_sig, trace_bkg \
            = contour.get_contour(img, threshold=_threshold, clip_iter=clip_iter,
                                  sigma_lower=sigma_lower, sigma_upper=sigma_upper)
    _img = img - trace_bkg
    circ_p = contour.Circle(trace[:,0], trace[:,1]).fit()

    # Use the coordinates to compute the radius of each pixel
    ny, nx = _img.shape
    x, y = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))
    circ_r, circ_theta = contour.Circle.polar(x, y, circ_p)

    # Construct 1D vectors with data sorted by radius:
    srt = numpy.argsort(circ_r.ravel())
    radius = circ_r.ravel()[srt]
    #   - Flux
    flux = img.ravel()[srt]
    #   - Cumulative sum of the flux to get the EE (growth) curve
    ee = numpy.cumsum(flux)

    # Iteratively determine a local background, if requested
    bg_r = None
    local_bg = 0.
    if local_bg_fac is not None:
        for i in range(local_iter):
            # Get the half-width of the EE curve
            hwhm = interpolate.interp1d(ee/ee[-1], radius)([0.5])[0]
            # Get the background in the selected regions
            bg_r = local_bg_fac*hwhm
            indx = radius > bg_r
            _local_bg = contour.get_bg(flux[indx], clip_iter=clip_iter, sigma_lower=sigma_lower,
                                       sigma_upper=sigma_upper)[0]
            # Subtract it from the image and flux
            _img -= _local_bg
            flux -= _local_bg
            # Recompute the EE curve
            ee = numpy.cumsum(flux)
            # Add it to the total
            local_bg += _local_bg
            print(f'local_bg: {local_bg}')

    return radius, ee



