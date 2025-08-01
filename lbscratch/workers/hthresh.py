# from lbscratch.workers.main import cli
# import click
# from omegaconf import OmegaConf
# import pyscilog
# pyscilog.init('lbscratch')
# log = pyscilog.get_logger('HTHRESH')

# from scabha.schema_utils import clickify_parameters
# from lbscratch.parser.schemas import schema

# # create default parameters from schema
# defaults = {}
# for key in schema.hthresh["inputs"].keys():
#     defaults[key] = schema.hthresh["inputs"][key]["default"]

# @cli.command(context_settings={'show_default': True})
# @clickify_parameters(schema.hthresh)
# def hthresh(**kw):
#     '''
#     Apply hard threshold to model and write out corresponding fits file
#     '''
#     defaults.update(kw)
#     opts = OmegaConf.create(defaults)
#     import time
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     pyscilog.log_to_file(f'hthresh_{timestamp}.log')

#     if opts.nworkers is None:
#         if opts.scheduler=='distributed':
#             opts.nworkers = opts.nband
#         else:
#             opts.nworkers = 1

#     OmegaConf.set_struct(opts, True)

#     # TODO - prettier config printing
#     print('Input Options:', file=log)
#     for key in opts.keys():
#         print('     %25s = %s' % (key, opts[key]), file=log)

#     import numpy as np
#     from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
#     from pfb.utils.fits import load_fits, set_wcs, save_fits

#     basename = f'{opts.output_filename}_{opts.product.upper()}'
#     mds_name = f'{basename}{opts.postfix}.mds.zarr'
#     mds = xds_from_zarr(mds_name, chunks={'band':1})[0]
#     nband = mds.nband
#     nx = mds.nx
#     ny = mds.ny

#     model = mds.get(opts.model_name).values

#     model = np.where(model < opts.threshold, 0.0, model)

#     radec = [mds.ra, mds.dec]
#     cell_rad = mds.cell_rad
#     cell_deg = np.rad2deg(cell_rad)
#     freq_out = mds.freq.data
#     ref_freq = np.mean(freq_out)
#     hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
#     hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

#     model_mfs = np.mean(model, axis=0)

#     save_fits(f'{basename}_threshold_model.fits', model, hdr)
#     save_fits(f'{basename}_threshold_model_mfs.fits', model_mfs, hdr_mfs)
