import os
import numpy as np
from scipy.stats import median_abs_deviation as mad
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
from pathlib import Path
import dask
import time
import concurrent.futures as cf
import xarray as xr
from lbscratch.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('lbscratch')
log = pyscilog.get_logger('CLIP_AMP')

from scabha.schema_utils import clickify_parameters
from lbscratch.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.clip_gain_amps["inputs"].keys():
    defaults[key] = schema.clip_gain_amps["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.clip_gain_amps)
def clip_gain_amps(**kw):
    '''
    Smooth and plot 1D gain solutions with a median filter
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    opts.nband = 1  # hack!!!
    OmegaConf.set_struct(opts, True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'bsmooth_{timestamp}.log')

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    gain_dir = str(Path(opts.gain).resolve())
    gain_dir = '::'.join(gain_dir.rsplit('/', 1))

    try:
        xds = xds_from_zarr(gain_dir)
        if not len(xds):
            raise ValueError(f'No data at {str(gain_dir)}')
    except Exception as e:
        raise(e)

    nscan = len(xds)
    ntime, nchan, nant, ndir, ncorr = xds[0].gains.data.shape
    if ndir > 1:
        raise NotImplementedError("DD gains not supported")

    ds, i = clip_amp(xds[0], opts.threshold, opts.window, 0)

    import ipdb; ipdb.set_trace()

    # with cf.ProcessPoolExecutor(max_workers=nscan) as executor:
    #     futures = []
    #     for i, ds in enumerate(xds):
    #         fut = executor.submit(clip_amp,
    #                               ds,
    #                               opts.threshold,
    #                               opts.window,
    #                               i)
    #     for fut in cf.as_completed(futures):
    #         ds, i = fut.result()
    #         xds[i] = ds
    #         print(f'Flagged scan {i}')

    # writes = xds_to_zarr(xds, gain_dir+'-flagged')
    # dask.compute(writes)



def clip_amp(ds, threshold, window, i):
    amp = np.abs(ds.gains.values)
    flags = ds.gain_flags.values
    freq = ds.gain_freq.values
    ntime, nchan, nant, ndir, ncorr  = amp.shape
    amp = np.where(flags, np.nan, amp)

    for p in range(nant):
        for f in range(nchan):
            # skip already flagged channels
            if flags[:, f].all():
                continue
            for c in range(ncorr):
                Ilow = np.maximum(0, f - window)
                Ihigh = np.minimum(Ilow + window, nchan)
                flag = flags[:, p, Ilow:Ihigh, 0, c] > 0
                data = amp[:, p, Ilow:Ihigh, 0, c]
                # check window is non-empty
                if flag.all():
                    flags[:, f, p, 0, c] = 1
                    continue
                # detrend
                data_time_med = np.nanmedian(data, axis=0, keepdims=True)
                data -= data_time_med
                median = np.nanmedian(data)
                madval = mad(data, axis=None, scale='normal', nan_policy='omit')
                diff = np.abs(data - median)
                flags[:, f, p, 0, c] = np.where(diff > threshold * mad, 1, 0)

    ds['gain_flags'] = (ds.gain_flags.dims, flags)
    return ds, i



