
from lbscratch.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('lbscratch')
log = pyscilog.get_logger('FLAGGER')

from scabha.schema_utils import clickify_parameters
from lbscratch.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.flagger["inputs"].keys():
    defaults[key] = schema.flagger["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.flagger)
def flagger(**kw):
    '''
    Flag based on non-constant auto-correlations
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'flagger_{timestamp}.log')
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    from multiprocessing.pool import ThreadPool
    import dask
    dask.config.set(pool=ThreadPool(opts.nthreads))
    # dask.config.set(scheduler='sync')
    import dask.array as da
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_to_storage_table as xds_to_table
    import numpy as np
    import xarray as xr
    from lbscratch.utils import flautos, set_flags, flags_at_edges
    from functools import partial
    from lbscratch.utils import make_flag_row


    if opts.mode == "autos":
        # we don't want to group by scan in this case, only chunk over freq axis
        xds = xds_from_ms(opts.ms,
                          columns=['DATA','FLAG','FLAG_ROW','ANTENNA1','ANTENNA2','TIME'],
                          chunks={'row':-1, 'chan': opts.chan_chunk},
                          group_cols=['FIELD_ID', 'DATA_DESC_ID'])  # , 'SCAN_NUMBER'

        xdso = []
        for ds in xds:
            flag = flautos(ds.DATA.data,
                           ds.FLAG.data,
                           ds.ANTENNA1.data,
                           ds.ANTENNA2.data,
                           ds.TIME.values,
                           sigma=opts.sigma)

            flag_row = make_flag_row(flag)

            dso = ds.assign(**{'FLAG': (('row','chan','corr'), flag),
                               'FLAG_ROW':(('row',), flag_row)})
            xdso.append(dso)

    elif opts.mode == "edges":
        # group by whatever, only chunk over row
        xds = xds_from_ms(opts.ms,
                          columns='FLAG',
                          chunks={'row':opts.row_chunk},
                          group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'])

        # 1419.8:1421.3 MHz <=> 2697:2705 chans
        I = np.zeros(xds[0].chan.size, dtype=bool)
        for idx in opts.franges.split(','):
            m = re.match('(-?\d+)?:(-?\d+)?', idx)
            ilow = int(m.group(1)) if m.group(1) is not None else None
            ihigh = int(m.group(2)) if m.group(2) is not None else None
            I[slice(ilow, ihigh)] = True

        for i, ds in enumerate(xds):
            flag = da.blockwise(set_flags, 'rfc',
                                flag, ds.FLAG.data,
                                I, 'f',
                                dtype=bool)

            dso = ds.assign(**{'FLAG': (('row','chan','corr'), flag)})
            xdso.append(dso)

    elif opts.mode == "persistent":
        xds = xds_from_ms(opts.ms,
                          columns=['FLAG','TIME','ANTENNA1','ANTENNA2'],
                          chunks={'row':-1, 'chan': opts.chan_chunk},
                          group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'])

        # get flags at scan boundaries
        tflags = []
        utimes = []
        for ds in xds:
            tflag, utime = flags_at_edges(ds.FLAG.data,
                                          ds.TIME.values)
            tflags.append(tflag)
            utimes.append(utime)

        tflags = dask.compute(tflags)[0]

        utimes = np.concatenate(utimes, axis=0)
        tflags = np.concatenate(tflags, axis=0)

        xds = xds_from_ms(opts.ms_target,
                          columns=['FLAG','TIME','ANTENNA1','ANTENNA2'],
                          chunks={'row':-1, 'chan': opts.chan_chunk},
                          group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'])

        tflags = da.from_array(tflags, chunks=(-1, -1, opts.chan_chunk, -1))

    writes = xds_to_table(xdso, opts.ms,
                          columns=["FLAG", "FLAG_ROW"],
                          rechunk=True)

    dask.compute(writes)





