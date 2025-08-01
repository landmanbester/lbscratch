import concurrent.futures as cf
import os
from pathlib import Path

import dask
import dask.array as da
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pyscilog
from scabha.schema_utils import clickify_parameters
import xarray as xr

from lbscratch.parser.schemas import schema
from lbscratch.utils import smooth_ant
from lbscratch.workers.main import cli

os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(1)
os.environ["NUMBA_NUM_THREADS"] = str(1)

dask.config.set(**{"array.slicing.split_large_chunks": False})
mpl.rcParams.update({"font.size": 18, "font.family": "serif"})

pyscilog.init("lbscratch")
log = pyscilog.get_logger("GSMOOTH")

# create default parameters from schema
defaults = {}
for key in schema.gsmooth["inputs"].keys():
    defaults[key] = schema.gsmooth["inputs"][key]["default"]


@cli.command(context_settings={"show_default": True})
@clickify_parameters(schema.gsmooth)
def gsmooth(**kw):
    """
    Smooth and plot 1D gain solutions with a median filter
    """
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    opts.nband = 1  # hack!!!
    import time

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f"gsmooth_{timestamp}.log")
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print("Input Options:", file=log)
    for key in opts.keys():
        print(f"     {key:>25} = {opts[key]}", file=log)

    gain_dir = Path(opts.gain_dir).resolve()

    try:
        xds = xds_from_zarr(f"{gain_dir!s}::{opts.gain_term}")
        if not len(xds):
            raise ValueError(f"No data at {gain_dir!s}::{opts.gain_term}")
    except Exception as e:
        raise (e)

    if opts.transfer_offset_from is not None:
        K = xds_from_zarr(f"{gain_dir!s}::{opts.transfer_offset_from}")

        ti = 0
        It = []
        for k, GK in enumerate(zip(xds, K, strict=False)):
            dsg, dsk = GK
            gain = np.zeros(dsg.gains.shape, dsg.gains.dtype)
            ntime = gain.shape[0]
            It.append(slice(ti, ti + ntime))
            ti += ntime
            try:
                assert dsk.params.shape[0] == 1
            except (AssertionError, AttributeError) as e:
                # TODO - interpolate offset to resolution of G
                raise NotImplementedError("Only scalar offset can be transfered atm") from e
            offset = dsk.params.values[0, 0, :, 0, 0]
            # K = exp(1.0j*(offset + 2pi*slope*freq))
            gain[:, :, :, :, 0] = dsg.gains.values[:, :, :, :, 0].copy() * np.exp(
                1.0j * offset[None, None, :, None]
            )
            offset = dsk.params.values[0, 0, :, 0, 2]
            gain[:, :, :, :, 1] = dsg.gains.values[:, :, :, :, 1].copy() * np.exp(
                1.0j * offset[None, None, :, None]
            )
            dsg = dsg.assign(
                **{"gains": (dsg.GAIN_AXES, da.from_array(gain, chunks=(-1, -1, -1, -1, -1)))}
            )
            xds[k] = dsg

            params = dsk.params.values.copy()
            params[:, :, :, :, 0] = 0.0
            params[:, :, :, :, 2] = 0.0
            freq = dsk.gain_freq.values.copy()[None, :, None, None, None]
            # this works because the offsets have been zeroed
            gain = np.exp(2.0j * np.pi * params[:, :, :, :, (1, 3)] * freq)
            dsk = dsk.assign(
                **{
                    "gains": (dsk.GAIN_AXES, da.from_array(gain, chunks=(-1, -1, -1, -1, -1))),
                    "params": (dsk.PARAM_AXES, da.from_array(params, chunks=(-1, -1, -1, -1, -1))),
                }
            )
            dsk.attrs["TYPE"] = "pure_delay"
            K[k] = dsk

        print(
            f"Writing pure delays (i.e. offset removed) to {gain_dir!s}/smoothed.qc::K", file=log
        )
        writes = xds_to_zarr(
            K,
            f"{gain_dir!s}/smoothed.qc::K",
            columns=(
                "gains",
                "gain_flags",
                "params",
                "param_flags",
                "jhj",
                "conv_iter",
                "conv_perc",
            ),
        )

        dask.compute(writes)

    if opts.do_smooth:
        xdso = []
        for ds in xds:
            print(f"Doing scan {ds.SCAN_NUMBER}", file=log)
            ntime, nchan, nant, ndir, ncorr = ds.gains.data.shape

            if opts.ref_ant == -1:
                ref_ant = nant - 1
            else:
                ref_ant = opts.ref_ant
            if nchan > 1:
                raise ValueError("Only time smoothing currently supported")
            if ndir > 1:
                raise ValueError("Only time smoothing currently supported")

            jhj = ds.jhj.values.copy().real
            g = ds.gains.values.copy()
            f = ds.gain_flags.values.copy()
            flag = np.logical_or(jhj == 0, f[:, :, :, :, None])
            jhj = np.where(flag, 0.0, jhj)

            # manual unwrap required?
            gamp = np.abs(g)
            gphase = np.angle(g * g[:, :, ref_ant].conj()[:, :, None])
            gphase = np.unwrap(gphase, axis=0, discont=0.9 * 2 * np.pi)

            t = ds.gain_time.values.copy()
            # scale t to lie in (0, 1)
            t -= t.min()
            t /= t.max()
            t += 0.1
            t *= 0.9 / t.max()

            samp = np.zeros_like(gamp)
            sphase = np.zeros_like(gphase)

            futures = []
            with cf.ProcessPoolExecutor(max_workers=opts.nthreads) as executor:
                for p in range(nant):
                    for c in range(ncorr):
                        w = jhj[:, 0, p, 0, c]
                        amp = gamp[:, 0, p, 0, c]
                        phase = gphase[:, 0, p, 0, c]
                        do_phase = p != ref_ant
                        future = executor.submit(
                            smooth_ant, amp, phase, w, t, p, c, do_phase=do_phase
                        )
                        futures.append(future)

                for future in cf.as_completed(futures):
                    sa, sp, p, c = future.result()
                    samp[:, 0, p, 0, c] = sa
                    sphase[:, 0, p, 0, c] = sp

            gs = samp * np.exp(1.0j * sphase)
            gs = da.from_array(gs, chunks=(-1, -1, -1, -1, -1))
            dso = ds.assign(**{"gains": (ds.GAIN_AXES, gs)})
            xdso.append(dso)
    else:
        xdso = xds

    print(f"Writing smoothed gains to {gain_dir!s}/smoothed.qc::{opts.gain_term}", file=log)
    writes = xds_to_zarr(
        xdso, f"{gain_dir!s}/smoothed.qc::{opts.gain_term}", columns=("gains", "gain_flags")
    )
    dask.compute(writes)

    if not opts.do_plots:
        print("Not doing plots", file=log)
        print("All done here", file=log)
        quit()

    # concatenate for plotting
    xds_concat = xr.concat(xds, dim="gain_time").sortby("gain_time")
    ntime, nchan, nant, ndir, ncorr = xds_concat.gains.data.shape
    xdso_concat = xr.concat(xdso, dim="gain_time").sortby("gain_time")
    jhj = xds_concat.jhj.values.copy().real
    g = xds_concat.gains.values.copy()
    gs = xdso_concat.gains.values.copy()
    f = xds_concat.gain_flags.values.copy()
    flag = np.logical_or(jhj == 0, f[:, :, :, :, None])
    jhj = np.where(flag, 0.0, jhj)

    # why manual unwrap sometimes required?
    gamp = np.abs(g)
    samp = np.abs(gs)
    gphase = np.angle(g * g[:, :, ref_ant].conj()[:, :, None])
    gphase = np.unwrap(gphase, axis=0, discont=0.9 * 2 * np.pi)
    sphase = np.angle(gs * gs[:, :, ref_ant].conj()[:, :, None])
    sphase = np.unwrap(sphase, axis=0, discont=0.9 * 2 * np.pi)
    # medvals0 = np.median(gphase[It[0], 0, :, 0, :], axis=0)
    # for I in It[1:]:
    #     medvals = np.median(gphase[I, 0, :, 0, :], axis=0)
    #     for p in range(nant):
    #         for c in range(ncorr):
    #             tmp = medvals[p, c] - medvals0[p, c]
    #             if np.abs(tmp) > 0.9*2*np.pi:
    #                 gphase[I, 0, p, 0, c] -= 2*np.pi*np.sign(tmp)

    t = xds_concat.gain_time.values.copy()

    # kernel = mat52()
    # theta0 = np.ones(3)
    # for p in range(nant):
    #     for c in range(ncorr):
    #         print(f" p = {p}, c = {c}")
    #         idx = np.where(jhj[:, 0, p, 0, c] > 0)[0]
    #         if idx.size < 2:
    #             continue
    #         w = jhj[:, 0, p, 0, c]
    #         amp = gamp[:, 0, p, 0, c]
    #         theta0[0] = np.std(amp[w!=0])
    #         theta0[1] = 0.25*t.max()
    #         _, mus, covs = emterp(theta0, t, amp, kernel, w=w, niter=opts.niter, nu=2)
    #         samp[:, 0, p, 0, c] = mus
    #         sampcov[:, 0, p, 0, c] = covs
    #         if p == ref_ant:
    #             continue
    #         phase = gphase[:, 0, p, 0, c]
    #         wp = w/samp[:, 0, p, 0, c]
    #         theta0[0] = np.std(phase[wp!=0])
    #         theta0[1] = 0.05*t.max()
    #         _, mus, covs = emterp(theta0, t, phase, kernel, w=wp, niter=opts.niter, nu=2)
    #         sphase[:, 0, p, 0, c] = mus
    #         sphasecov[:, 0, p, 0, c] = covs

    # set to NaN's for plotting
    gamp = np.where(jhj > 0, gamp, np.nan)
    gphase = np.where(jhj > 0, gphase, np.nan)

    samp = np.where(jhj > 0, samp, np.nan)
    sphase = np.where(jhj > 0, sphase, np.nan)

    print("Plotting results", file=log)
    futures = []
    with cf.ProcessPoolExecutor(max_workers=opts.nthreads) as executor:
        for p in range(nant):
            for c in range(ncorr):
                ga = gamp[:, 0, p, 0, c]
                sa = samp[:, 0, p, 0, c]
                gp = gphase[:, 0, p, 0, c]
                sp = sphase[:, 0, p, 0, c]
                w = jhj[:, 0, p, 0, c]
                future = executor.submit(plot_ant, ga, sa, gp, sp, w, t, p, c, opts, gain_dir)
                futures.append(future)

        for future in cf.as_completed(futures):
            future.result()

    print("All done here.", file=log)


def plot_ant(gamp, samp, gphase, sphase, jhj, t, p, c, opts, gain_dir):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 18))
    fig.suptitle(f"Antenna {p}, corr {c}", fontsize=24)
    sigma = np.where(jhj > 0, 1.0 / np.sqrt(jhj), np.nan)
    ax[0].errorbar(t, gamp, sigma, fmt="xr", label="raw")
    ax[0].plot(t, samp, "ko", label="smooth")
    ax[0].legend()
    ax[0].set_xlabel("time")

    sigmap = sigma / samp
    ax[1].errorbar(t, np.rad2deg(gphase), sigmap, fmt="xr", label="raw")
    ax[1].plot(t, np.rad2deg(sphase), "ko", label="smooth")
    ax[1].legend()
    ax[1].set_xlabel("time")

    fig.tight_layout()
    name = f"{gain_dir!s}/{opts.gain_term}_Antenna{p}corr{c}.png"
    plt.savefig(name, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot {name}", file=log)
