import re

from omegaconf import OmegaConf
import pyscilog

from lbscratch.workers.main import cli

pyscilog.init("lbscratch")
log = pyscilog.get_logger("FLEDGES")

from scabha.schema_utils import clickify_parameters

from lbscratch.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.fledges["inputs"].keys():
    defaults[key] = schema.fledges["inputs"][key]["default"]


@cli.command(context_settings={"show_default": True})
@clickify_parameters(schema.fledges)
def fledges(**kw):
    """
    Apply a frequency mask
    """
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f"fledges_{timestamp}.log")
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print("Input Options:", file=log)
    for key in opts.keys():
        print(f"     {key:>25} = {opts[key]}", file=log)

    from multiprocessing.pool import ThreadPool

    import dask

    dask.config.set(pool=ThreadPool(opts.nthreads))
    import dask.array as da
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms import xds_to_storage_table as xds_to_table
    import numpy as np

    xds = xds_from_ms(
        opts.ms,
        columns="FLAG",
        chunks={"row": opts.row_chunk},
        group_cols=["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"],
    )
    freq = xds_from_table(f"{opts.ms}::SPECTRAL_WINDOW")[0].CHAN_FREQ.values
    # assuming single spectral window
    freq = freq.squeeze()

    # 1397.4:1398.2 <=> 2590:2595
    # 1419.8:1421.3 <=> 2697:2705
    freq_mask = np.zeros(xds[0].chan.size, dtype=bool)
    for idx in opts.franges.split(","):
        ilow, ihigh = parse_range_with_units(idx, freq)
        freq_mask[slice(ilow, ihigh)] = True

    freq_mask = da.from_array(freq_mask, chunks=-1)
    xdso = []
    for ds in xds:
        flag = ds.FLAG.data
        flag = da.blockwise(set_flags, "rfc", flag, "rfc", freq_mask, "f", dtype=bool)

        dso = ds.assign(**{"FLAG": (("row", "chan", "corr"), flag)})
        xdso.append(dso)

    writes = xds_to_table(xdso, opts.ms, columns="FLAG", rechunk=True)

    dask.compute(writes)


def set_flags(flag, freq_mask):
    flag[:, freq_mask, :] = True
    return flag


def parse_range_with_units(range_str, freq=None):
    # Pattern to match float values with optional units
    import numpy as np

    pattern = r"^(?:([-?\d.]+)([a-zA-Z]+)?)?(?::(?:([-?\d.]+)([a-zA-Z]+)?)?)?$"
    match = re.match(pattern, range_str.strip())

    if not match:
        raise ValueError(f"Invalid range format: {range_str}")

    start_val, start_unit, end_val, end_unit = match.groups()

    # Empty string units should be converted to None
    start_unit = start_unit.lower() if start_unit else None
    end_unit = end_unit.lower() if end_unit else None

    cdict = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
    if start_unit is None:
        # indices passed in
        start_idx = int(start_val) if start_val is not None else None
    else:
        if freq is None:
            raise ValueError("No coordinates passed in")
        if start_val is not None:
            start = float(start_val) * cdict[start_unit]
            start_idx = int(np.abs(freq - start).argmin())
        else:
            start_idx = None

    if end_unit is None:
        # indices passed in
        end_idx = int(end_val) if end_val is not None else None
    else:
        if end_val is not None:
            end = float(end_val) * cdict[end_unit]
            end_idx = int(np.abs(freq - end).argmin())
        else:
            end_idx = None

    # Create and return the slice object
    return start_idx, end_idx
