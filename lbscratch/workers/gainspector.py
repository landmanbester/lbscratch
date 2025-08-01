from omegaconf import ListConfig, OmegaConf
import pyscilog

from lbscratch.workers.main import cli

pyscilog.init("lbscratch")
log = pyscilog.get_logger("GAINSPECTOR")

from scabha.schema_utils import clickify_parameters

from lbscratch.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.gainspector["inputs"].keys():
    defaults[key] = schema.gainspector["inputs"][key]["default"]


@cli.command(context_settings={"show_default": True})
@clickify_parameters(schema.gainspector)
def gainspector(**kw):
    """
    Plot effective gains produced my QuartiCal
    """
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    opts.nband = 1
    import time

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f"gainspector_{timestamp}.log")

    from glob import glob

    if opts.gain_dir is not None:
        gt = glob(opts.gain_dir)
        try:
            assert len(gt) > 0
            opts.gain_dir = gt
        except Exception as err:
            raise ValueError(f"No gain table  at {opts.gain_dir}") from err

    if opts.nworkers is None:
        if opts.scheduler == "distributed":
            opts.nworkers = opts.nband
        else:
            opts.nworkers = 1

    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print("Input Options:", file=log)
    for key in opts.keys():
        print(f"     {key:>25} = {opts[key]}", file=log)

    if not isinstance(opts.gain_dir, list) and not isinstance(opts.gain_dir, ListConfig):
        opts.gain_dir = [opts.gain_dir]
    OmegaConf.set_struct(opts, True)

    # import matplotlib as mpl
    # mpl.rcParams.update({'font.size': 10, 'font.family': 'serif'})
    import dask
    import numpy as np

    dask.config.set(**{"array.slicing.split_large_chunks": False})
    from daskms.experimental.zarr import xds_from_zarr
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import xarray as xr

    Gs = []
    for gain in opts.gain_dir:
        # why do we distinguish between :: and /?
        try:
            G = xds_from_zarr(f"{gain}::{opts.gain_term}")
        except Exception as err:
            raise ValueError(f"Failed to load gains at {gain}::{opts.gain_term}") from err
        for g in G:
            Gs.append(g)

    if opts.join_times:
        Gs = [xr.concat(Gs, dim="gain_time")]

    for s, G in enumerate(Gs):
        gain = G.gains.sortby("gain_time")
        try:
            flag = G.gain_flags.sortby("gain_time")
        except Exception:
            flag = None
            print("No gain flags found", file=log)
        ntime, nchan, nant, ndir, ncorr = gain.shape
        if opts.ref_ant is not None:
            if opts.ref_ant == -1:
                ref_ant = nant - 1
            else:
                ref_ant = opts.ref_ant
            gref = gain[:, :, ref_ant]
        else:
            gref = np.ones((ntime, nchan, ndir, ncorr))
        for c in [0, 1]:
            ntot = ntime + nchan
            tlength = int(np.ceil(11 * ntime / ntot))
            fig, axs = plt.subplots(nrows=nant, ncols=1, figsize=(10, nant * tlength))
            for i, ax in enumerate(axs.ravel()):
                if i < nant:
                    if opts.mode == "ampphase":
                        g = np.abs(gain.values[:, :, i, 0, c])
                    elif opts.mode == "reim":
                        g = np.real(gain.values[:, :, i, 0, c])
                    else:
                        raise ValueError(f"Unknown mode {opts.mode}")
                    if flag is not None:
                        f = flag.values[:, :, i, 0]
                        It, If = np.where(f)
                        g[It, If] = np.nan

                    if opts.vmina is not None:
                        glow = opts.vmina
                        ghigh = opts.vmaxa
                    else:
                        gmed = np.nanmedian(g)
                        gstd = np.nanstd(g)
                        glow = gmed - opts.vlow * gstd
                        ghigh = gmed + opts.vhigh * gstd

                    im = ax.imshow(g, cmap="inferno", interpolation=None, vmin=glow, vmax=ghigh)
                    ax.set_title(f"Antenna: {i}")
                    ax.axis("off")

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("bottom", size="10%", pad=0.01)
                    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                    cb.outline.set_visible(False)
                    cb.ax.tick_params(length=0.1, width=0.1, labelsize=10.0, pad=0.1)
                else:
                    ax.axis("off")

            fig.tight_layout()
            name = "real" if opts.mode == "reim" else "abs"
            plt.savefig(
                opts.output_filename + f"_corr{c}_scan{s}_{name}.png", dpi=100, bbox_inches="tight"
            )
            plt.close()

            fig, axs = plt.subplots(nrows=nant, ncols=1, figsize=(10, nant * tlength))

            for i, ax in enumerate(axs.ravel()):
                if i < nant:
                    if opts.mode == "ampphase":
                        g = gain.values[:, :, i, 0, c] * gref[:, :, 0, c].conj()
                        g = np.unwrap(np.unwrap(np.angle(g), axis=0), axis=1)
                    elif opts.mode == "reim":
                        g = np.imag(gain.values[:, :, i, 0, c])
                    else:
                        raise ValueError(f"Unknown mode {opts.mode}")

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

                    im = ax.imshow(g, cmap="inferno", interpolation=None, vmin=glow, vmax=ghigh)
                    ax.set_title(f"Antenna: {i}")
                    ax.axis("off")

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("bottom", size="10%", pad=0.01)
                    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                    cb.outline.set_visible(False)
                    cb.ax.tick_params(length=0.5, width=0.5, labelsize=10, pad=0.5)
                else:
                    ax.axis("off")

            fig.tight_layout()

            name = "imag" if opts.mode == "reim" else "phase"
            plt.savefig(
                opts.output_filename + f"_corr{c}_scan{s}_{name}.png", dpi=100, bbox_inches="tight"
            )
            plt.close()
            try:
                jhj = G.jhj.sortby("gain_time")
                fig, axs = plt.subplots(nrows=nant, ncols=1, figsize=(10, nant * tlength))
                for i, ax in enumerate(axs.ravel()):
                    if i < nant:
                        g = jhj.values[:, :, i, 0, c]
                        if flag is not None:
                            f = flag.values[:, :, i, 0]
                            It, If = np.where(f)
                            g[It, If] = np.nan

                        im = ax.imshow(np.abs(g), cmap="inferno", interpolation=None)
                        ax.set_title(f"Antenna: {i}")
                        ax.axis("off")

                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("bottom", size="10%", pad=0.01)
                        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                        cb.outline.set_visible(False)
                        cb.ax.tick_params(length=0.5, width=0.5, labelsize=10, pad=0.5)
                    else:
                        ax.axis("off")

                fig.tight_layout()

                plt.savefig(
                    opts.output_filename + f"_corr{c}_scan{s}_jhj.png",
                    dpi=100,
                    bbox_inches="tight",
                )
                plt.close()
            except Exception:
                continue
