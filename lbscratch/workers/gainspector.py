# flake8: noqa
from contextlib import ExitStack
from lbscratch.workers.main import cli
import click
from omegaconf import OmegaConf
from omegaconf import ListConfig
import pyscilog
pyscilog.init('lbscratch')
log = pyscilog.get_logger('GAINSPECTOR')

from scabha.schema_utils import clickify_parameters
from lbscratch.parser.schemas import schema

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.gainspector)
def gainspector(**kw):
    '''
    Plot effective gains produced by QuartiCal
    '''
    opts = OmegaConf.create(kw)

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'gainspector_{timestamp}.log')

    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)


    # import matplotlib as mpl
    # mpl.rcParams.update({'font.size': 10, 'font.family': 'serif'})
    import numpy as np
    import dask
    dask.config.set(**{'array.slicing.split_large_chunks': False})
    from daskms.experimental.zarr import xds_from_zarr
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import xarray as xr

    # complex gains
    try:
        G = xds_from_zarr('::'.join(opts.complex_gain.rsplit('/', 1)),
                          chunks=None)
    except:
        raise ValueError(f'Failed to load complex gains at {opts.complex_gain}')
    # net gains
    try:
        N = xds_from_zarr('::'.join(opts.net_gain.rsplit('/', 1)),
                          chunks=None)
    except:
        raise ValueError(f'Failed to load net gains at {opts.net_gain}')

    for gds, nds in zip(G, N):
        try:
            assert gds.gains.data.shape == nds.gains.data.shape
        except Exception as e:
            raise ValueError("Gains have inconsistent shape")

    if opts.join_times:
        G = [xr.concat(G, dim='gain_time')]
        N = [xr.concat(N, dim='gain_time')]

    for s, gds, nds in enumerate(zip(G, N)):
        ntime, nchan, nant, ndir, ncorr = gds.gains.shape
        for p in range(nant):
            for c in [0,-1]:



            for i, ax in enumerate(axs.ravel()):
                if i < nant:
                    if opts.mode == 'ampphase':
                        g = (gain.values[:, :, i, 0, c] *
                             gref[:, :, 0, c].conj())
                        g = np.unwrap(np.unwrap(np.angle(g), axis=0), axis=1)
                    elif opts.mode == 'reim':
                        g = np.imag(gain.values[:, :, i, 0, c])
                    else:
                        raise ValueError(f'Unknown mode {opts.mode}')

                    if flag is not None:
                        f = flag.values[:, :, i, 0]
                        It, If = np.where(f)
                        g[It, If] = np.nan

                    if opts.vminp is not None:
                        glow = opts.vminp
                        ghigh = opts.vmaxp
                    else:
                        gmed = np.nanmedian(g)
                        gstd = np.nanstd(g)
                        glow = gmed - opts.vlow * gstd
                        ghigh = gmed + opts.vhigh * gstd

                    im = ax.imshow(g, cmap='inferno', interpolation=None,
                                   vmin=glow, vmax=ghigh)
                    ax.set_title(f"Antenna: {i}")
                    ax.axis('off')

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("bottom", size="10%", pad=0.01)
                    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                    cb.outline.set_visible(False)
                    cb.ax.tick_params(length=0.5, width=0.5,
                                      labelsize=10, pad=0.5)
                else:
                    ax.axis('off')

            fig.tight_layout()

            name = 'imag' if opts.mode == 'reim' else 'phase'
            plt.savefig(opts.output_filename + f"_corr{c}_scan{s}_{name}.png",
                        dpi=100, bbox_inches='tight')
            plt.close()
            try:
                jhj = G.jhj.sortby('gain_time')
                fig, axs = plt.subplots(nrows=nant, ncols=1,
                                        figsize=(10, nant*tlength))
                for i, ax in enumerate(axs.ravel()):
                    if i < nant:
                        g = jhj.values[:, :, i, 0, c]
                        if flag is not None:
                            f = flag.values[:, :, i, 0]
                            It, If = np.where(f)
                            g[It, If] = np.nan

                        im = ax.imshow(np.abs(g), cmap='inferno',
                                       interpolation=None)
                        ax.set_title(f"Antenna: {i}")
                        ax.axis('off')

                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("bottom", size="10%", pad=0.01)
                        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                        cb.outline.set_visible(False)
                        cb.ax.tick_params(length=0.5, width=0.5,
                                          labelsize=10, pad=0.5)
                    else:
                        ax.axis('off')

                fig.tight_layout()

                plt.savefig(opts.output_filename + f"_corr{c}_scan{s}_jhj.png",
                            dpi=100, bbox_inches='tight')
                plt.close()
            except Exception as e:
                continue


def plot_gains(gds, nds, ref_ant, p, c):
    gain = gds.gains.values
    net = nds.gains.values
    resid = gain - net
    flag = gds.gain_flags.values
    It, Inu = np.where(flag)

    ntime, nchan = gain.shape
    ntot = ntime + nchan
    flength = 10
    tlength = int(np.ceil(flength * ntime/ntot))

    # real
    fig, axs = plt.subplots(nrows=1, ncols=2,
                            figsize=(flength, 2*tlength))

    residr =resid.real
    residr[It, If] = np.nan


    gmed = np.nanmedian(g)
    gstd = np.nanstd(g)
    glow = gmed - opts.vlow * gstd
    ghigh = gmed + opts.vhigh * gstd

    im = ax.imshow(g,
                    cmap='inferno', interpolation=None,
                    vmin=glow, vmax=ghigh)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="10%", pad=0.01)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0.1, width=0.1,
                        labelsize=10.0, pad=0.1)

    fig.tight_layout()
    name = 'real' if opts.mode == 'reim' else 'abs'
    fig.savefig(opts.output_filename + f"_corr{c}_scan{s}_{name}.png",
                dpi=100, bbox_inches='tight')

    # phases
    fig, axs = plt.subplots(nrows=nant, ncols=1,
                            figsize=(10, nant*tlength))
