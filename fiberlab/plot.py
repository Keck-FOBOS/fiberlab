
from IPython import embed

import numpy
from matplotlib import pyplot, ticker

def frd_plot(files, nas=None, labels=None, colors=None, theta_lim=None, dtheta_lim=None,
             fratio_lim=None, dfratio_lim=None, ofile=None):

    for f in files:
        if not f.exists():
            raise FileNotFoundError(f'{f} does not exist!')

    n = len(files)
    if any([o is not None and len(o) != n for o in [files, nas, labels, colors]]):
        raise ValueError('Mismatch in lists')

    if colors is None:
        colors = [f'C{i}' for i in range(n)]

    # Assume index of refraction for air is 1.
#    alpha = numpy.degrees(numpy.arcsin(nas))
    fratio = None if nas is None else 1/2/numpy.asarray(nas)

    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(1.5*w,1.5*h))

    tdt_ax = fig.add_axes([0.1, 0.14, 0.36, 0.36])
    tdt_ax.minorticks_on()
    tdt_ax.tick_params(which='both', direction='in', top=True, right=True)
    tdt_ax.grid(True, which='major', color='0.8', zorder=0, ls='-', lw=0.5)
    tdt_ax.text(-0.15, 0.5, r'$\Delta\theta$ [deg]',
                ha='center', va='center', transform=tdt_ax.transAxes, rotation='vertical')
    tdt_ax.text(0.5, -0.12, r'$\theta$ [deg]',
                ha='center', va='center', transform=tdt_ax.transAxes)
    f_ax = fig.add_axes([0.59, 0.5, 0.36, 0.36])
    f_ax.minorticks_on()
    f_ax.tick_params(which='both', direction='in', top=True, right=True)
    f_ax.grid(True, which='major', color='0.8', zorder=0, ls='-', lw=0.5)
    f_ax.xaxis.set_major_formatter(ticker.NullFormatter())
    f_ax.text(-0.15, 0.5, r'Output f-ratio',
              ha='center', va='center', transform=f_ax.transAxes, rotation='vertical')
    fdf_ax = fig.add_axes([0.59, 0.14, 0.36, 0.36])
    fdf_ax.minorticks_on()
    fdf_ax.tick_params(which='both', direction='in', top=True, right=True)
    fdf_ax.grid(True, which='major', color='0.8', zorder=0, ls='-', lw=0.5)
    fdf_ax.text(-0.15, 0.5, r'$\delta$ f-ratio',
                ha='center', va='center', transform=fdf_ax.transAxes, rotation='vertical')
    fdf_ax.text(0.5, -0.12, r'Input f-ratio',
                ha='center', va='center', transform=fdf_ax.transAxes)

    _theta_lim = [100, -1]
    _dtheta_lim = [100, -1]
    _fratio_lim = [100, -1]
    _dfratio_lim = [100, -1]

    for i in range(n):
        db = numpy.genfromtxt(str(files[i]), dtype=str)
        theta = db[:,4].astype(float)
        dtheta = db[:,5].astype(float)

        indx = theta > 1.
        theta = theta[indx]
        dtheta = dtheta[indx]
        srt = numpy.argsort(theta)
        theta = theta[srt]
        dtheta = dtheta[srt]

        in_fratio = 1 / 2 / numpy.tan(numpy.radians(theta))
        out_fratio = 1 / 2 / numpy.tan(numpy.radians(theta + dtheta/2))

        dfratio = in_fratio/out_fratio - 1.

        _theta_lim = [min(_theta_lim[0], numpy.amin(theta)),
                     max(_theta_lim[1], numpy.amax(theta))]
        _dtheta_lim = [min(_dtheta_lim[0], numpy.amin(dtheta)),
                      max(_dtheta_lim[1], numpy.amax(dtheta))]
        _fratio_lim = [min(_fratio_lim[0], numpy.amin(numpy.append(in_fratio, out_fratio))),
                      max(_fratio_lim[1], numpy.amax(numpy.append(in_fratio, out_fratio)))]
        _dfratio_lim = [min(_dfratio_lim[0], numpy.amin(dfratio)),
                       max(_dfratio_lim[1], numpy.amax(dfratio))]

        gpm = numpy.ones_like(in_fratio, dtype=bool) if fratio is None else in_fratio > fratio[i]
        bpm = numpy.logical_not(gpm)
        if any(bpm):
            indx = numpy.where(bpm)[0]
            bpm[indx[0]-1] = True

        tdt_ax.scatter(theta[gpm], dtheta[gpm],
                       marker='.', s=20, lw=0, c=colors[i], zorder=6)
        tdt_ax.plot(theta[gpm], dtheta[gpm], c=colors[i], lw=0.5, zorder=5)
        tdt_ax.plot(theta[bpm], dtheta[bpm], c=colors[i], lw=0.5, zorder=4, ls='--')

        f_ax.scatter(in_fratio[gpm], out_fratio[gpm],
                     marker='.', s=20, lw=0, c=colors[i], zorder=6)
        f_ax.plot(in_fratio[gpm], out_fratio[gpm], c=colors[i], lw=0.5, zorder=5)
        f_ax.plot(in_fratio[bpm], out_fratio[bpm], c=colors[i], lw=0.5, zorder=4, ls='--')

        fdf_ax.scatter(in_fratio[gpm], dfratio[gpm],
                       marker='.', s=20, lw=0, c=colors[i], zorder=6)
        fdf_ax.plot(in_fratio[gpm], dfratio[gpm], c=colors[i], lw=0.5, zorder=5)
        fdf_ax.plot(in_fratio[bpm], dfratio[bpm], c=colors[i], lw=0.5, zorder=4, ls='--')
        if labels is not None:
            tdt_ax.text(-0.1, 2.0 - i*0.15, labels[i], color=colors[i],
                        transform=tdt_ax.transAxes, ha='left', va='top', weight='bold')

    if theta_lim is None:
        theta_lim = _theta_lim
    if dtheta_lim is None:
        dtheta_lim = _dtheta_lim
    if fratio_lim is None:
        fratio_lim = _fratio_lim
    if dfratio_lim is None:
        dfratio_lim = _dfratio_lim

    tdt_ax.set_xlim(theta_lim)
    tdt_ax.set_ylim(dtheta_lim)

    f_ax.set_xlim(fratio_lim[::-1])
    f_ax.set_ylim(fratio_lim[::-1])
    f_ax.plot(fratio_lim, fratio_lim, color='k', zorder=2, ls='--', lw=0.5)

    fdf_ax.set_xlim(fratio_lim[::-1])
    fdf_ax.set_ylim(dfratio_lim)

    if ofile is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(ofile, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)

