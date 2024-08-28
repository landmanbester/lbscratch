from lbscratch.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('lbscratch')
log = pyscilog.get_logger('CHANFLAGS')

from scabha.schema_utils import clickify_parameters
from lbscratch.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.chanflags["inputs"].keys():
    defaults[key] = schema.chanflags["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.chanflags)
def chanflags(**kw):
    '''
    Channel flag summary
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'restimator_{timestamp}.log')
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    import numpy as np
    import dask
    import dask.array as da
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from pathlib import Path
    import matplotlib.pyplot as plt

    ms_path = Path(opts.ms).resolve()
    ms_name = str(ms_path)
    xds = xds_from_ms(ms_name,
                      columns=['FLAG'],
                      group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'],
                      chunks={'row': -1, 'chan': 256, 'corr': -1})
    spw = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0]
    freq = spw.CHAN_FREQ.data[0]

    counts = []
    nrow = 0
    for ds in xds:
        nflag = ~ds.FLAG.data
        nrow += nflag.shape[0]
        counts.append(da.sum(nflag, axis=(0, -1))[:, None])

    freq, counts = dask.compute(freq, counts)

    count = np.stack(counts, axis=1)
    count = count.sum(axis=1)

    _, nchan, ncorr = nflag.shape

    mask = count[:, 0] > 0
    if mask.any():
        unflagged_freqs = freq[mask]
        print(f'First and last unflagged freqs = {unflagged_freqs[0]} and {unflagged_freqs[-1]}', file=log)
        print(f'Corresponding inices = {mask.argmax()} and {nchan - mask[::-1].argmax()}', file=log)
    else:
        print(f'All freqs are flagged', file=log)

    if opts.oname is not None:
        oname = Path(opts.oname).resolve()
        oname.parent.mkdir(parents=True, exist_ok=True)
        oname = str(oname)
    else:  # put next to MS by default
        idx = ms_name.rfind('.')
        suffix = ms_name[idx:]
        oname = ms_name.rstrip(suffix) + '_chanflags.png'

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xticks(freq[0:-1:opts.channels_per_xtick], labelsize=6)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.plot(freq, count.astype(np.float32)/(nrow*ncorr), 'xr')

    # import ipdb; ipdb.set_trace()
    fig.savefig(oname)
