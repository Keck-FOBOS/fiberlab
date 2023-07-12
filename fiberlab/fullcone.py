"""
Module with methods used to analyze and plot full-cone test images.

.. include:: ../include/links.rst
"""
import warnings

from IPython import embed

import numpy
from scipy import interpolate, ndimage
from matplotlib import pyplot, ticker

from astropy.stats import sigma_clip

from . import contour
from .io import bench_image
from . import util


# TODO: Why is this a function?
def default_threshold():
    """
    The default threshold to use for defining the contour used to determine the
    center of the output beam.
    """
    return 3.0


# TODO: Change the name of this function to just 'farfield'?
def fullcone_farfield_output(img_file, bkg_file=None, pixelsize=None, distance=None,
                             plot_file=None, snr_img=False, window=None, box=None, **kwargs):
    """
    Analyze an output far-field image for a full-cone input beam.

    Args:
        img_file (:obj:`str`, `Path`_):
            Image file
        bkg_file (:obj:`str`, `Path`_, optional):
            Background image to subtract
        pixelsize (:obj:`float`, optional):
            Size of the *native* image pixels in mm (i.e., before any boxcar
            average; see ``box``).
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
        **kwargs:
            Parameters passed directly to :class:`EECurve` used to derive the
            encircled-energy curve.

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

    # Analyze the image
    ee = EECurve(img_nobg, **kwargs)

    r_units, circ_r = contour.convert_radius(ee.circ_r, pixelsize=_pixelsize, distance=distance)
    radius = contour.convert_radius(ee.radius, pixelsize=_pixelsize, distance=distance)[1]
    model_radius = contour.convert_radius(ee.model_radius, pixelsize=_pixelsize,
                                          distance=distance)[1]

    if plot_file is not None:
        fullcone_farfield_output_plot(img_file, ee.img, ee.model_img,
                                      ee.trace, ee.circ_p, radius, ee.flux, ee.ee/ee.ee_norm,
                                      ee.model_radius, ee.model_flux, snr_img=snr_img,
                                      img_sig=ee.level/ee.threshold,
                                      bkg_lim=ee.bkg_lim, r_units=r_units,
                                      window=window, pixelsize=_pixelsize, distance=distance,
                                      ofile=None if plot_file == 'show' else plot_file)

    return ee


def fullcone_farfield_output_plot(img_file, img, model, trace, circ_p, radius, flux,
                                  ee, model_radius, model_flux, snr_img=False,
                                  img_sig=None, bkg_lim=None,
                                  r_units='pix', window=None, pixelsize=None, distance=None,
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
        level (:obj:`float`, optional):
            The level in the image that corresponds to the S/N threshold.
        r_umits (:obj:`str`, optional):
            The units of the radius vector.
        window (:obj:`float`, optional):
            Limit the plotted image regions to this times the best-fitting
            contour radius.  If None, the full image is shown.
        ofile (:obj:`str`, `Path`_, optional):
            Name of the QA file to produce.  If ``'show'``, no file is produced,
            but the QA plot is displayed in a matplotlib window.
    """

    right = interpolate.interp1d(ee, radius)(0.85)
    print(f'EE85: {right:.2f}')

    xc, yc, rc = circ_p
    rc *= pixelsize
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

    image_lim = contour.growth_lim(numpy.append(img, model), 0.99, fac=1.2)
    resid_lim = contour.growth_lim(resid, 0.90, fac=1.0, midpoint=0.)
    rfac = window
    if bkg_lim is not None:
        rfac = 1.1 * bkg_lim[0] if rfac is None else max(1.1 * bkg_lim[0], rfac)
        if len(bkg_lim) == 2:
            rfac = max(1.1 * bkg_lim[1], rfac)
    radius_lim = [0, numpy.amax(radius) if rfac is None else rfac*rc]

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
        snr_lim = contour.growth_lim(img/img_sig, 0.99, fac=1.2)
        imgplt = ax.imshow(img/img_sig, origin='lower', interpolation='nearest', cmap=cmap,
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
    ax.axvline(rc, color='C3', lw=1, ls='--', zorder=6)
    if bkg_lim is not None:
        ax.axvline(bkg_lim[0]*rc, color='0.5', lw=1, ls='--', zorder=6)
        if len(bkg_lim) == 2:
            ax.axvline(bkg_lim[1]*rc, color='0.5', lw=1, ls='--', zorder=6)

    ax.text(0.5, -0.08, f'Radius [{r_units}]', ha='center', va='center', transform=ax.transAxes)

    ax.text(-0.05, -0.12, f'Directory: {img_file.parent.name}', ha='left', va='center',
            transform=ax.transAxes)
    ax.text(-0.05, -0.17, f'File: {img_file.name}', ha='left', va='center',
            transform=ax.transAxes)
    ax.text(-0.05, -0.22, f'EE90 radius: {right:.2f}', ha='left', va='center',
            transform=ax.transAxes)

    axt = ax.twinx()
    axt.set_xlim(radius_lim)
    axt.spines['right'].set_color('0.3')
    axt.tick_params(which='both', axis='y', direction='in', colors='0.3')
    axt.yaxis.label.set_color('0.3')
    axt.plot(radius, ee, color='0.3', lw=0.5, zorder=5)
    axt.set_ylabel('Enclosed Energy')
    axt.axhline(1.0, color='0.5', lw=1, ls='--', zorder=6)

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

    inp_radius, _, inp_ee, _, _, _ \
            = ee_curve(inp_data, threshold=threshold, clip_iter=clip_iter, sigma_lower=sigma_lower,
                       sigma_upper=sigma_upper, local_iter=local_iter, local_bg_fac=local_bg_fac)
    inp_hwhm = interpolate.interp1d(inp_ee/inp_ee[-1], inp_radius)([0.5])[0]
    inp_flux = numpy.mean(inp_ee[inp_radius > 2*inp_hwhm])

    out_radius, _, out_ee, _, _, _ \
            = ee_curve(out_data, threshold=threshold, clip_iter=clip_iter, sigma_lower=sigma_lower,
                       sigma_upper=sigma_upper, local_iter=local_iter, local_bg_fac=local_bg_fac)

    out_hwhm = interpolate.interp1d(out_ee/out_ee[-1], out_radius)([0.5])[0]
    out_flux = numpy.mean(out_ee[out_radius > 2*out_hwhm])

    return inp_flux, out_flux, out_flux/inp_flux


# Removed for now:
#        local_iter (:obj:`float`, optional):
#            Iteratively measure and update a local background using the region
#            defined by ``bkg_lim``.  The background is measured using the same
#            clipping iterations and sigma as used when measuring the image
#            contour used to find the beam center (see ``clip_iter``,
#            ``sigma_lower``, and ``sigma_upper``).  If None, no local background
#            is measured and subtracted from the EE curve.
#
#        local_iter_sig (:obj:`float`, optional):
#            Number of iterations to perform for the local background
#            subtraction.
#                 , local_iter=None, local_iter_sig=3.):
class EECurve:
    """
    Construct an encircled energy curve provided an output image with a single
    output beam.

    Args:
        img (`numpy.ndarray`_):
            Image data to analze.
        mask (`numpy.ndarray`_, optional):
            Boolean mask image (True=masked pixel).  Must be the same shape as
            ``img``.  If None, all pixels in ``img`` are considered valid.
        circ_p (`numpy.ndarray`_, optional):
            Parameters of the circle used to define where the output beam is
            located in the image.  Shape must be ``(3,)``, providing (1) the
            center along the image 2nd axis (following the numpy convention of
            this being the ``x`` coordinate), (2) the center along the 1st axis
            (numpy ``y`` axis), and (3) a fiducial radius of the spot.  If None,
            these parameters are determine by fitting an image contour.  If
            provided, ``smooth`` and ``threshold`` are ignored.
        smooth (:obj:`float`, optional):
            When using an image contour to find the output beam center, first
            smooth the image using a Gaussian kernel with this (circular) sigma.
            If None, ``img`` is not smoothed before the contour is determined.
        threshold (:obj:`float`, optional):
            The threshold in units of image background standard deviation used
            to set the contour level.  If None and ``circ_p`` is not provided,
            the default value is set by :func:`default_threshold`.
        clip_iter (:obj:`int`, optional):
            Number of clipping iterations to perform when measuring the
            background and standard deviation in the input image.  This is used
            only when setting the image contour level to find the beam center.
            If None, no clipping iterations are performed.  The purpose of this
            clipping is to remove most/all pixels with light from the spot.
        sigma_lower (:obj:`float`, optional):
            Sigma rejection used for measurements below the mean.  Must be
            provided (or left at the default) if ``clip_iter`` is not None.
            Typically this number should be large: pixels at relatively low
            values should be background pixels, where as pixels above the mean
            have light from the output spot.
        sigma_upper (:obj:`float`, optional):
            Sigma rejection used for measurements above the mean.  Must be
            provided (or left at the default) if ``clip_iter`` is not None.
            Typically this number should be small: pixels at relatively low
            values should be background pixels, where as pixels above the mean
            have light from the output spot.
        bkg_lim (:obj:`float`, array-like, optional):
            Perform a +/- sigma rejection in a background region defined by
            this object.  If a single value is provided, all pixels beyond the
            multiple of the beam radius, defined by the last element of
            ``circ_p``, is included in the rejection.  Instead, a list or numpy
            array can be used to define an inner (first element) and outer (last
            element) multiple for the radius.  If None, no additional background
            region rejection is performed.
        bkg_lim_sig (:obj:`float`, optional):
            Number of sigma for the background rejection performed based on
            ``bkg_lim``.
    """
    def __init__(self, img, mask=None, circ_p=None, smooth=None, threshold=None, clip_iter=10,
                 sigma_lower=100., sigma_upper=3., bkg_lim=None, bkg_lim_sig=3.):

        self.img = img.copy()
        self.gpm = None if mask is None else numpy.logical_not(mask)

        # Get the bounding contour of the image spot
        if circ_p is None:
            # Image should already have a nominal background subtracted
            _img = self.img
            if smooth is not None:
                _img = self.img.copy()
                if mask is not None:
                    _img[mask] = 0.
                _img = ndimage.gaussian_filter(_img, sigma=smooth)

            self.threshold = default_threshold() if threshold is None else threshold
            self.level, self.trace, self.trace_sig, self.bkg \
                    = contour.get_contour(numpy.ma.MaskedArray(_img, mask=mask),
                                          threshold=self.threshold, clip_iter=clip_iter,
                                          sigma_lower=sigma_lower, sigma_upper=sigma_upper)
            # Fit a circle to the contour
            self.circ_p = contour.Circle(self.trace[:,0], self.trace[:,1]).fit()
            self.img -= self.bkg
        else:
            self.threshold = None
            self.level = None
            self.trace = None
            self.trace_sig = None
            self.bkg = 0.
            self.circ_p = circ_p

        # Use the coordinates to compute the radius of each pixel
        ny, nx = self.img.shape
        self.x, self.y = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))
        self.circ_r, circ_theta = contour.Circle.polar(self.x, self.y, self.circ_p)

        # Reject pixels at large radius outside of the main spot and adjust the
        # background
        self.bkg_lim = bkg_lim
        if self.bkg_lim is not None:
            # Select the pixels in the background region
            if isinstance(self.bkg_lim, list):
                indx = (self.circ_r > self.circ_p[2] * self.bkg_lim[0]) \
                            & (self.circ_r < self.circ_p[2] * self.bkg_lim[1])
            else:
                indx = self.circ_r > self.circ_p[2] * self.bkg_lim
            if self.gpm is not None:
                indx &= self.gpm
            # Clip the pixel values
            clipped = sigma_clip(self.img[indx], sigma=bkg_lim_sig)
            # Get the adjusted background
            bkg_adj = numpy.ma.median(clipped)
            # Add it to the previous background determination
            self.bkg += bkg_adj
            # And subtract it from the image
            self.img -= bkg_adj
            # Add the clipped pixels to the mask
            if mask is None:
                mask = numpy.zeros(self.img.shape, dtype=bool)
                mask[indx] = numpy.ma.getmaskarray(clipped).copy()
            else:
                mask[indx] |= numpy.ma.getmaskarray(clipped)
            self.gpm = numpy.logical_not(mask)

            # Also mask pixels at *all* radii beyond the inner radius (but
            # they're not included in the background).  This removes them from
            # consideration in the EE curve, but excludes them from the
            # background calculation.
            if isinstance(self.bkg_lim, list):
                indx = self.gpm & (self.circ_r > self.circ_p[2] * self.bkg_lim[0])
                clipped = sigma_clip(self.img[indx], sigma=3.)
                mask[indx] |= numpy.ma.getmaskarray(clipped)
                self.gpm = numpy.logical_not(mask)

        # Only select the unmasked pixels when constructing the EE curve
        if self.gpm is None:
            self.radius = self.circ_r.ravel()
            self.flux = self.img.ravel()
        else:
            self.radius = self.circ_r[self.gpm].ravel()
            self.flux = self.img[self.gpm].ravel()

        # Construct 1D vectors with the data sorted by radius:
        srt = numpy.argsort(self.radius)
        self.radius = self.radius[srt]
        #   - Flux
        self.flux = self.flux[srt]
        #   - Cumulative sum of the flux to get the EE (growth) curve
        self.ee = numpy.cumsum(self.flux)
        #   - Get the normalization
        self.get_ee_norm()
        #   - Build model
        self.build_model()

    def get_ee_norm(self):
        if self.bkg_lim is None:
            self.ee_norm = self.ee[-1]
        elif isinstance(self.bkg_lim, list):
            indx = (self.radius > self.bkg_lim[0]*self.circ_p[2]) \
                        & (self.radius < self.bkg_lim[1]*self.circ_p[2])
            self.ee_norm = numpy.median(self.ee[indx])
        else:
            indx = self.radius > self.circ_p[2] * self.bkg_lim
            self.ee_norm = numpy.median(self.ee[indx])

    def build_model(self):
        # Create a smoothed version of the flux
        self.smooth_flux = contour.iterative_filter(self.flux, 301, 2) #, clip_iter=10, sigma=5.)
        # Create a down-sampled set of radii for the model
        self.model_radius = numpy.linspace(0,max(self.radius),500)[1:]
        # Sample the smoothed flux to get the model flux
        self.model_flux = interpolate.interp1d(self.radius, self.smooth_flux,
                                               fill_value='extrapolate')(self.model_radius)
        # Sample the EE
        self.model_ee \
                = interpolate.interp1d(self.radius, self.ee, bounds_error=False,
                                       fill_value=(self.ee[0], self.ee[-1]))(self.model_radius)
        # Create the model image
        self.model_img = interpolate.interp1d(self.model_radius, self.model_flux,
                                bounds_error=False,
                                fill_value=(self.model_flux[0], self.model_flux[-1]))(self.circ_r)


def calculate_fratio(z0_radius, z0_ee, z1_radius, z1_ee, sep, smooth=False, z0_bkg_lim=None,
                     z1_bkg_lim=None):
    """
    Given the normalized EE curves from two images separated by a known
    distance, calculate the distance from each image to the origin of the output
    beam and the beam focal ratio.

    Image z0 is taken closer to the output beam, and image z1 is taken further away.

    Args:
        z0_radius (`numpy.ndarray`_):
            Radius in mm from the center of the spot in image z0.
        z0_ee (`numpy.ndarray`_):
            Normalized encircled energy for the spot in image z0.
        z1_radius (`numpy.ndarray`_):
            Radius in mm from the center of the spot in image z1.
        z1_ee (`numpy.ndarray`_):
            Normalized encircled energy for the spot in image z1.
        sep (:obj:`float`):
            Separation between the two images in mm along the output axis.
        smooth (:obj:`bool`, optional):
            Smooth the raw data before interpolating the radius at a fixed set
            of EE values.
        z0_bkg_lim (:obj:`float`, optional):
            Radius in mm from the center of the spot at which the background
            flux was determined in image z0.
        z1_bkg_lim (:obj:`float`, optional):
            Radius in mm from the center of the spot at which the background
            flux was determined in image z1.

    Returns:
        :obj:`tuple`: A set of ten objects: (1) the EE samples, (2-3) the radii
        at which the spot reaches the sampled EE for z0 and z1, respectively,
        (4-5) the distances between the origin of the output beam and the image
        as measured at each EE sample for z0 and z1, respectively, (6-7) the
        "median" distance estimate using 0.2 < EE < 0.9 for image z0 and z1,
        respectively, (8-9) the focal ratios calculated using the median
        distances and the measured radii for image z0 and z1, respectively, and
        (10) a boolean indicating if the calculation was successful.
    """
    # Smooth the EE curves, if requested
    _z0_ee = contour.iterative_filter(z0_ee, 301, 2) if smooth else z0_ee
    _z1_ee = contour.iterative_filter(z1_ee, 301, 2) if smooth else z1_ee

    # Find the index of the last value where the normalized EE is still
    # increasing
    z0_last = numpy.where(numpy.diff(_z0_ee) < 0)[0][0]
    z1_last = numpy.where(numpy.diff(_z1_ee) < 0)[0][0]

    # Force the last index to be at a radius less than the beginning of the
    # background region
    if z0_bkg_lim is not None:
        z0_last = min(z0_last, numpy.where(z0_radius > z0_bkg_lim)[0][0])
    if z1_bkg_lim is not None:
        z1_last = min(z1_last, numpy.where(z1_radius > z1_bkg_lim)[0][0])

    # Get a uniform set of EE samples from 1% to 99%
    ee_sample = numpy.linspace(0.01, 0.99, 99)
    empty = numpy.full(ee_sample.size, -1, dtype=float)
    success = True
    try:
        ee_r_z0 = interpolate.interp1d(_z0_ee[:z0_last], z0_radius[:z0_last])(ee_sample)
    except:
        warnings.warn('Error interpolating raw EE data for z0 image.')
        success = False
        ee_r_z0 = empty

    try:
        ee_r_z1 = interpolate.interp1d(_z1_ee[:z1_last], z1_radius[:z1_last])(ee_sample)
    except:
        warnings.warn('Error interpolating raw EE data for z1 image.')
        success = False
        ee_r_z1 = empty

    if not success:
        return ee_sample, ee_r_z0, ee_r_z1, empty, empty, -1., -1., empty, empty, success

    # Calculate the distance for all samples
    z0_distance = sep/(ee_r_z1/ee_r_z0-1)
    z1_distance = sep/(1-ee_r_z0/ee_r_z1)

    # Use the median ratio of the radii over EE values from 0.2 to 0.9 to
    # calculate a "median" distance
    indx = (ee_sample >= 0.2) & (ee_sample <= 0.9)
    med_z0_distance = sep/(numpy.median(ee_r_z1[indx]/ee_r_z0[indx])-1)
    med_z1_distance = sep/(1-numpy.median(ee_r_z0[indx]/ee_r_z1[indx]))

    # Use the median distance to calculate the f/#
    ee_fratio_z0 = med_z0_distance / 2 / ee_r_z0
    ee_fratio_z1 = med_z1_distance / 2 / ee_r_z1

    return ee_sample, ee_r_z0, ee_r_z1, z0_distance, z1_distance, \
            med_z0_distance, med_z1_distance, ee_fratio_z0, ee_fratio_z1, success


