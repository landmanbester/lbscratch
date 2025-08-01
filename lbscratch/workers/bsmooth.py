import concurrent.futures as cf
import os
from pathlib import Path
import time

import dask
import dask.array as da
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
from omegaconf import OmegaConf
import pyscilog
from scabha.schema_utils import clickify_parameters

from lbscratch.parser.schemas import schema
from lbscratch.utils import smooth_ant
from lbscratch.workers.main import cli

os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(1)
os.environ["NUMBA_NUM_THREADS"] = str(1)

mpl.rcParams.update({"font.size": 18, "font.family": "serif"})

pyscilog.init("lbscratch")
log = pyscilog.get_logger("BSMOOTH")

# create default parameters from schema
defaults = {}
for key in schema.bsmooth["inputs"].keys():
    defaults[key] = schema.bsmooth["inputs"][key]["default"]


@cli.command(context_settings={"show_default": True})
@clickify_parameters(schema.bsmooth)
def bsmooth(**kw):
    """
    Smooth and plot 1D gain solutions with a median filter
    """
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    opts.nband = 1  # hack!!!
    OmegaConf.set_struct(opts, True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f"bsmooth_{timestamp}.log")

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

    nscan = len(xds)
    ntime, nchan, nant, ndir, ncorr = xds[0].gains.data.shape
    if ntime > 1:
        raise ValueError("Only freq smoothing currently supported")
    if ndir > 1:
        raise ValueError("Only freq smoothing currently supported")

    if opts.ref_ant == -1:
        ref_ant = nant - 1
    else:
        ref_ant = opts.ref_ant

    nscan = len(xds)  # assumed
    if opts.per_scan:
        bamp = np.zeros((nscan, ntime, nchan, nant, ndir, ncorr), dtype=np.float64)
        bphase = np.zeros((nscan, ntime, nchan, nant, ndir, ncorr), dtype=np.float64)
        samp = np.zeros((nscan, ntime, nchan, nant, ndir, ncorr), dtype=np.float64)
        sphase = np.zeros((nscan, ntime, nchan, nant, ndir, ncorr), dtype=np.float64)
        wgt = np.zeros((nscan, ntime, nchan, nant, ndir, ncorr), dtype=np.float64)
        for i, ds in enumerate(xds):
            # copy to avoid mutating on disc data after write
            freq = ds.gain_freq.values.copy()
            jhj = ds.jhj.values.real.copy()
            g = ds.gains.values.copy()
            f = ds.gain_flags.values.copy()
            # the smoothing coordinate needs to be normalised to lie between (0, 1)
            fmin = freq.min()
            fmax = freq.max()
            nu = (freq - fmin) / (fmax - fmin)
            nu += 0.1
            nu *= 0.9 / nu.max()
            # ensure consistent flag <-> jhj
            flag = np.logical_or(jhj == 0, f[:, :, :, :, None])
            jhj = np.where(flag, 0.0, jhj)

            amp = np.abs(g)
            jhj = np.where(amp < opts.reject_amp_thresh, jhj, 0)

            phase = np.angle(g) - np.angle(g[:, :, ref_ant])[:, :, None]
            jhj = np.where(np.abs(phase) < np.deg2rad(opts.reject_phase_thresh), jhj, 0)

            bamp[i] = np.where(jhj > 0, amp, np.nan)
            bphase[i] = np.where(jhj > 0, phase, np.nan)
            wgt[i] = jhj

            print(f"Smoothing scan {i}", file=log)
            futures = []
            with cf.ProcessPoolExecutor(max_workers=opts.nthreads) as executor:
                for p in range(nant):
                    for c in range(ncorr):
                        w = jhj[0, :, p, 0, c]
                        do_phase = p != ref_ant
                        future = executor.submit(
                            smooth_ant,
                            amp[0, :, p, 0, c],
                            phase[0, :, p, 0, c],
                            w,
                            nu,
                            p,
                            c,
                            do_phase=do_phase,
                            niter=opts.nreweight,
                            dof=opts.dof0,
                            sigman_min=opts.sigman_min,
                        )
                        futures.append(future)

                for future in cf.as_completed(futures):
                    sa, sp, p, c = future.result()
                    # print(f" p = {p}, c = {c}", file=log)
                    samp[i, 0, :, p, 0, c] = sa
                    sphase[i, 0, :, p, 0, c] = sp

            bpass = samp[i] * np.exp(1.0j * sphase[i])
            bpass = da.from_array(bpass, chunks=(-1, -1, -1, -1, -1))
            xds[i] = ds.assign(**{"gains": (ds.GAIN_AXES, bpass)})

        print(f"Writing smoothed gains to {gain_dir!s}/smoothed.qc::{opts.gain_term}", file=log)
        writes = xds_to_zarr(xds, f"{gain_dir!s}/smoothed.qc::{opts.gain_term}", columns="ALL")

        dask.compute(writes)

        futures = []
        with cf.ProcessPoolExecutor(max_workers=opts.nthreads) as executor:
            for p in range(nant):
                for c in range(ncorr):
                    future = executor.submit(
                        plot_ant_scan,
                        bamp[:, 0, :, p, 0, c],
                        samp[:, 0, :, p, 0, c],
                        bphase[:, 0, :, p, 0, c],
                        sphase[:, 0, :, p, 0, c],
                        wgt[:, 0, :, p, 0, c],
                        freq,
                        p,
                        c,
                        bphase[:, 0, :, ref_ant, 0, c],
                        opts,
                        gain_dir,
                    )
                    futures.append(future)

            for future in cf.as_completed(futures):
                future.result()

    else:
        # we assume the freq range is the same per ds
        freq = xds[0].gain_freq.values.copy()
        # the smoothing coordinate needs to be normalised to lie between (0, 1)
        fmin = freq.min()
        fmax = freq.max()
        nu = (freq - fmin) / (fmax - fmin)
        nu += 0.1
        nu *= 0.9 / nu.max()

        bamp = np.zeros((ntime, nchan, nant, ndir, ncorr), dtype=np.float64)
        bphase = np.zeros((ntime, nchan, nant, ndir, ncorr), dtype=np.float64)
        wgt = np.zeros((ntime, nchan, nant, ndir, ncorr), dtype=np.float64)
        for ds in xds:
            jhj = np.abs(ds.jhj.values.copy())
            g = ds.gains.values.copy()
            f = ds.gain_flags.values.copy()
            # ensure consistent flag <-> jhj
            flag = np.logical_or(jhj == 0, f[:, :, :, :, None])
            jhj = np.where(flag, 0.0, jhj)

            amp = np.abs(g)
            jhj = np.where(amp < opts.reject_amp_thresh, jhj, 0)

            phase = np.angle(g) - np.angle(g[:, :, ref_ant])[:, :, None]
            jhj = np.where(np.abs(phase) < np.deg2rad(opts.reject_phase_thresh), jhj, 0)

            # remove slope BEFORE averaging
            if opts.detrend:
                for p in range(nant):
                    for c in range(ncorr):
                        idx = np.where(jhj[0, :, p, 0, c] > 0)[0]
                        if idx.size < 2:
                            continue
                        # enforce zero offset and slope
                        w = np.sqrt(jhj[0, idx, p, 0, c])  # polyfit convention
                        y = phase[0, idx, p, 0, c]
                        f = freq[idx]
                        coeffs = np.polyfit(f, y, 1, w=w)
                        phase[0, idx, p, 0, c] -= np.polyval(coeffs, f)

                        # TODO - move slope into K

            # to compute weighted sum
            bamp += amp * jhj
            bphase += phase * jhj
            wgt += jhj

        print("Smoothing over all scans", file=log)
        mask = wgt > 0
        bamp[mask] = bamp[mask] / wgt[mask]
        bphase[mask] = bphase[mask] / wgt[mask]

        samp = np.zeros_like(bamp)
        sphase = np.zeros_like(bamp)
        futures = []
        with cf.ProcessPoolExecutor(max_workers=opts.nthreads) as executor:
            for p in range(nant):
                for c in range(ncorr):
                    w = wgt[0, :, p, 0, c]
                    amp = bamp[0, :, p, 0, c]
                    phase = bphase[0, :, p, 0, c]
                    do_phase = p != ref_ant
                    future = executor.submit(
                        smooth_ant,
                        amp,
                        phase,
                        w,
                        nu,
                        p,
                        c,
                        do_phase=do_phase,
                        niter=opts.nreweight,
                        dof=opts.dof0,
                        sigman_min=opts.sigman_min,
                    )
                    futures.append(future)

            for future in cf.as_completed(futures):
                sa, sp, p, c = future.result()
                print(f" p = {p}, c = {c}", file=log)
                samp[0, :, p, 0, c] = sa
                sphase[0, :, p, 0, c] = sp

        bpass = samp * np.exp(1.0j * sphase)
        bpass = da.from_array(bpass, chunks=(-1, -1, -1, -1, -1))
        for i, ds in enumerate(xds):
            xds[i] = ds.assign(**{"gains": (ds.GAIN_AXES, bpass)})

        print(f"Writing smoothed gains to {gain_dir!s}/smoothed.qc::{opts.gain_term}", file=log)
        writes = xds_to_zarr(xds, f"{gain_dir!s}/smoothed.qc::{opts.gain_term}", columns="ALL")

        bpass = dask.compute(bpass, writes)[0]

        # set to NaN's for plotting
        bamp = np.where(wgt > 0, bamp, np.nan)
        bphase = np.where(wgt > 0, bphase, np.nan)

        samp = np.abs(bpass)
        sphase = np.angle(bpass)
        samp = np.where(wgt > 0, samp, np.nan)
        sphase = np.where(wgt > 0, sphase, np.nan)

        # load the original data for comparitive plotting
        # need to redo since xds was overwritten
        try:
            xds = xds_from_zarr(f"{gain_dir!s}::{opts.gain_term}")
        except Exception as e:
            raise (e)

        # we want to avoid repeatedly reading these from disk while plotting
        flags = {}
        gains = {}
        for ds in xds:
            jf = ds.jhj.values.real == 0.0
            f = ds.gain_flags.values
            flag = np.logical_or(jf, f[:, :, :, :, None])
            gain = ds.gains.values
            for p in range(nant):
                for c in range(ncorr):
                    try:
                        s = ds.SCAN_NUMBER
                    except AttributeError:
                        s = "all"
                    gains.setdefault(f"{p}_{c}", {})
                    gains[f"{p}_{c}"][s] = gain[0, :, p, 0, c]
                    flags.setdefault(f"{p}_{c}", {})
                    flags[f"{p}_{c}"][s] = flag[0, :, p, 0, c]

        freq = xds[0].gain_freq / 1e6  # MHz
        futures = []
        with cf.ProcessPoolExecutor(max_workers=opts.nthreads) as executor:
            for p in range(nant):
                for c in range(ncorr):
                    future = executor.submit(
                        plot_ant,
                        bamp[0, :, p, 0, c],
                        samp[0, :, p, 0, c],
                        bphase[0, :, p, 0, c],
                        sphase[0, :, p, 0, c],
                        gains[f"{p}_{c}"],
                        flags[f"{p}_{c}"],
                        freq,
                        p,
                        c,
                        gains[f"{opts.ref_ant}_{c}"],
                        opts,
                        gain_dir,
                    )

                    futures.append(future)

            for future in cf.as_completed(futures):
                future.result()

    print("All done here", file=log)


def plot_ant(bamp, samp, bphase, sphase, gains, flags, xcoord, p, c, ref_gain, opts, gain_dir):
    # ForkedPdb().set_trace()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 18))
    fig.suptitle(f"Antenna {p}, corr {c}", fontsize=24)

    majorLocator = MultipleLocator(256)
    majorFormatter = FormatStrFormatter("%d")

    for s in gains.keys():
        flag = flags[s]
        gain = gains[s]
        tamp = np.abs(gain)
        tphase = np.angle(gain) - np.angle(ref_gain[s])
        tamp[flag] = np.nan
        tphase[flag] = np.nan
        # print(tamp.shape, tphase.shape)
        ax[0].plot(xcoord, tamp, label=f"scan-{s}", alpha=0.5, linewidth=1)
        ax[1].plot(xcoord, np.rad2deg(tphase), label=f"scan-{s}", alpha=0.5, linewidth=1)

    ax[0].plot(xcoord, bamp, "k", label="inf", linewidth=1)
    ax[0].plot(xcoord, samp, "r", label="smooth", linewidth=1)
    ax2 = ax[0].twiny()
    ax2.set_xticks(range(xcoord.size))
    ax2.xaxis.set_major_locator(majorLocator)
    ax2.xaxis.set_major_formatter(majorFormatter)
    ax[0].legend()
    ax[0].set_xlabel("freq / [MHz]")

    ax[1].plot(xcoord, np.rad2deg(bphase), "k", label="inf", linewidth=1)
    ax[1].plot(xcoord, np.rad2deg(sphase), "r", label="smooth", linewidth=1)
    ax2 = ax[1].twiny()
    ax2.set_xticks(range(xcoord.size))
    ax2.xaxis.set_major_locator(majorLocator)
    ax2.xaxis.set_major_formatter(majorFormatter)
    ax[1].legend()
    ax[1].set_xlabel("freq / [MHz]")

    fig.tight_layout()
    name = f"{gain_dir!s}/{opts.gain_term}_Antenna{p}corr{c}.png"
    plt.savefig(name, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot {name}", file=log)


def plot_ant_scan(bamp, samp, bphase, sphase, wgt, xcoord, p, c, ref_phase, opts, gain_dir):
    # ForkedPdb().set_trace()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 18))
    fig.suptitle(f"Antenna {p}, corr {c}", fontsize=24)

    nscan = bamp.shape[0]
    rng = np.random.default_rng(42)
    for s in range(nscan):
        color = rng.random(size=3)

        w = wgt[s]

        # raw
        amp = np.where(w > 0, bamp[s], np.nan)
        phase = np.where(w > 0, bphase[s] - ref_phase[s], np.nan)
        ax[0].plot(xcoord, amp, c=color, label=f"scan-{s}", alpha=0.5, linewidth=1)
        ax[1].plot(xcoord, np.rad2deg(phase), c=color, alpha=0.5, linewidth=1)

        # smooth
        amp = np.where(w > 0, samp[s], np.nan)
        phase = np.where(w > 0, sphase[s] - ref_phase[s], np.nan)
        ax[0].plot(xcoord, amp, c=color, alpha=0.5, linewidth=1)
        ax[1].plot(xcoord, np.rad2deg(phase), c=color, alpha=0.5, linewidth=1)

    ax[0].legend()
    ax[0].set_xlabel("freq / [MHz]")
    ax[1].set_xlabel("freq / [MHz]")

    fig.tight_layout()
    name = f"{gain_dir!s}/{opts.gain_term}_Antenna{p}corr{c}.png"
    plt.savefig(name, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot {name}", file=log)
