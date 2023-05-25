from contextlib import ExitStack
from pfb.workers.experimental import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('FLAGGER')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.flagger["inputs"].keys():
    defaults[key] = schema.flagger["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.flagger)
def flagger(**kw):
    '''
    Apply basic flagging operations
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'flagger_{timestamp}.log')
    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:


        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _flagger(**opts)

def _flagger(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    from multiprocessing.pool import ThreadPool
    import dask
    dask.config.set(pool=ThreadPool(opts.nthreads))
    import dask.array as da
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_to_storage_table as xds_to_table
    import numpy as np
    import xarray as xr
    from pfb.utils.misc import chunkify_rows

    xds = xds_from_ms(opts.ms, columns='FLAG,ANTENNA1,ANTENNA2,TIME',
                      chunks={'row':-1, 'chan': opts.chan_chunk},
                      group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'])



    def row_to_tbl(data, ant1, ant2, time):
        nant = np.maximum(ant1, ant2) + 1
        nbl = nant * (nant - 1)/2 + nant  # including autos
        utime = np.unique(time)
        ntime = utime.size
        dims = data.shape[1:]
        out_data = np.zeros((ntime, nbl) + dims, dtype=data.dtype)
        P, Q = np.tril_indices(nant)

        row_chunks, rbin_idx, rbin_counts = chunkify_rows(time, 1)

        for t in range(ntime):
            for row in range(rbin_idx[t],
                             rbin_idx[t] + rbin_counts[t]):
                p = ant1[row]
                q = ant2[row]
                b = index_where(p, q, P, Q)
                out_data[t, b] = data[row]
        return out_data


def index_where(p, q, P, Q):
    return ((P==p) & (Q==q)).argmax()
