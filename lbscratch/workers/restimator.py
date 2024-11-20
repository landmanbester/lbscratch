from contextlib import ExitStack
from lbscratch.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('lbscratch')
log = pyscilog.get_logger('RESTIMATOR')

from scabha.schema_utils import clickify_parameters
from lbscratch.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.restimator["inputs"].keys():
    defaults[key] = schema.restimator["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.restimator)
def restimator(**kw):
    '''
    Find best reference antenna based on flagged percentages
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

    ms_path = Path(opts.ms).resolve()
    ms_name = str(ms_path)
    xds = xds_from_ms(ms_name,
                      group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'],
                      chunks={'row': -1, 'chan': -1, 'corr': -1})
    ants = xds_from_table(ms_name + '::ANTENNA')[0].NAME.values

    ant1 = xds[0].ANTENNA1.values
    ant2 = xds[0].ANTENNA2.values
    nant = np.maximum(ant1.max(), ant2.max()) + 1
    flags = np.zeros(nant)
    counts = np.zeros(nant)
    for ds in xds:
        flag = ds.FLAG.values
        ant1 = ds.ANTENNA1.values
        ant2 = ds.ANTENNA2.values
        for p in range(nant):
            I = np.where((ant1==p) | (ant2==p))[0]
            flags[p] += np.sum(flag[I])
            counts[p] += flag[I].size

    best_ant = 0
    best_percent = 100
    for p in range(nant):
        flag_percent = flags[p]*100/counts[p]
        if flag_percent < best_percent:
            best_ant = p
            best_percent = flag_percent

        print(f'{flag_percent} percent flagged for antenna {p} with name {ants[p]}', file=log)

    print(f'Best reference antenna based on flagged percentage is antenna {best_ant}')
