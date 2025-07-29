# # flake8: noqa
# from contextlib import ExitStack
# from lbscratch.workers.main import cli
# from functools import partial
# import click
# from omegaconf import OmegaConf
# import pyscilog
# pyscilog.init('lbscratch')
# log = pyscilog.get_logger('HPASS_CLEAN')

# from scabha.schema_utils import clickify_parameters
# from lbscratch.parser.schemas import schema

# # create default parameters from schema
# defaults = {}
# for key in schema.hpass_clean["inputs"].keys():
#     defaults[key] = schema.hpass_clean["inputs"][key]["default"]

# @cli.command(context_settings={'show_default': True})
# @clickify_parameters(schema.hpass_clean)
# def hpass_clean(**kw):
#     '''
#     High pass clean.
#     '''
#     defaults.update(kw)
#     opts = OmegaConf.create(defaults)
#     import time
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     pyscilog.log_to_file(f'hpass_clean_{timestamp}.log')

#     if opts.nworkers is None:
#         if opts.scheduler=='distributed':
#             opts.nworkers = opts.nband
#         else:
#             opts.nworkers = 1

#     OmegaConf.set_struct(opts, True)

#     with ExitStack() as stack:
#         # numpy imports have to happen after this step
#         from lbscratch import set_client
#         set_client(opts, stack, log, scheduler=opts.scheduler)

#         # TODO - prettier config printing
#         print('Input Options:', file=log)
#         for key in opts.keys():
#             print('     %25s = %s' % (key, opts[key]), file=log)

#         return _hpass_clean(**opts)

# def _hpass_clean(**kw):
#     opts = OmegaConf.create(kw)
#     OmegaConf.set_struct(opts, True)

#     import numpy as np
#     import xarray as xr
#     import numexpr as ne
#     import dask
#     import dask.array as da
#     from dask.distributed import performance_report
#     from pfb.utils.fits import load_fits, set_wcs, save_fits, data_from_header
#     from pfb.utils.misc import setup_image_data
#     from pfb.deconv.hogbom import hogbom
#     from pfb.deconv.clark import clark
#     from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
#     from pfb.operators.hessian import hessian
#     from pfb.opt.pcg import pcg
#     from pfb.operators.hessian import hessian_xds
#     from pfb.operators.psf import _hessian_reg_psf
#     from scipy import ndimage

#     basename = f'{opts.output_filename}_{opts.product.upper()}'

#     mds_name = f'{basename}{opts.postfix}.mds.zarr'

#     mds = xds_from_zarr(mds_name, chunks={'band':1})[0]
#     nband = mds.nband
#     nx = mds.nx
#     ny = mds.ny

#     # stitch dirty/psf in apparent scale
#     output_type = mds.CLEAN_MODEL.dtype
#     model = mds.CLEAN_MODEL.values
#     model_mfs = np.mean(model, axis=0)


#     if opts.threshold is None:
#         threshold = opts.sigmathreshold * rms
#     else:
#         threshold = opts.threshold

#     print("Iter %i: peak residual = %f, rms = %f" % (0, rmax, rms), file=log)
#     for k in range(opts.nmiter):
#         if opts.algo.lower() == 'clark':
#             print("Running Clark", file=log)
#             x, status = clark(residual, psf, psfo,
#                               threshold=threshold,
#                               gamma=opts.gamma,
#                               pf=opts.peak_factor,
#                               maxit=opts.clark_maxit,
#                               subpf=opts.sub_peak_factor,
#                               submaxit=opts.sub_maxit,
#                               verbosity=opts.verbose,
#                               report_freq=opts.report_freq)
#         elif opts.algo.lower() == 'hogbom':
#             print("Running Hogbom", file=log)
#             x, status = hogbom(residual, psf,
#                                threshold=threshold,
#                                gamma=opts.gamma,
#                                pf=opts.peak_factor,
#                                maxit=opts.hogbom_maxit,
#                                verbosity=opts.verbose,
#                                report_freq=opts.report_freq)
#         else:
#             raise ValueError(f'{opts.algo} is not a valid algo option')

#         # if clean has stalled or not converged do flux mop step
#         if opts.mop_flux and status:
#             print(f"Mopping flux at iter {k+1}", file=log)
#             mask = (np.any(x, axis=0) | np.any(model, axis=0))
#             if opts.dirosion:
#                 struct = ndimage.generate_binary_structure(2, opts.dirosion)
#                 mask = ndimage.binary_dilation(mask, structure=struct)
#                 mask = ndimage.binary_erosion(mask, structure=struct)
#             mask = mask[None, :, :]
#             x = pcg(lambda x: mask * psfo(mask*x), mask * residual, x,
#                     tol=opts.cg_tol, maxit=opts.cg_maxit,
#                     minit=opts.cg_minit, verbosity=opts.cg_verbose,
#                     report_freq=opts.cg_report_freq,
#                     backtrack=opts.backtrack)

#         model += x

#         print("Getting residual", file=log)
#         convimage = hess(model)
#         ne.evaluate('dirty - convimage', out=residual,
#                     casting='same_kind')
#         ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
#                     casting='same_kind')

#         # save_fits(opts.output_filename + f'_residual_mfs{k}.fits',
#         #           residual_mfs, hdr_mfs)
#         # save_fits(opts.output_filename + f'_model_mfs{k}.fits',
#         #           np.mean(model, axis=0), hdr_mfs)
#         # save_fits(opts.output_filename + f'_convim_mfs{k}.fits',
#         #           np.sum(convimage, axis=0), hdr_mfs)

#         rms = np.std(residual_mfs[~np.any(model, axis=0)])
#         rmax = np.abs(residual_mfs).max()

#         print(f"Iter {k+1}: peak residual = {rmax}, rms = {rms}", file=log)

#         if opts.threshold is None:
#             threshold = opts.sigmathreshold * rms
#         else:
#             threshold = opts.threshold

#         if rmax <= threshold:
#             print("Terminating because final threshold has been reached",
#                   file=log)
#             break

#     print("Saving results", file=log)
#     if opts.update_mask:
#         mask = np.any(model > rms, axis=0)
#         if opts.dirosion:
#             struct = ndimage.generate_binary_structure(2, opts.dirosion)
#             mask = ndimage.binary_dilation(mask, structure=struct)
#             mask = ndimage.binary_erosion(mask, structure=struct)
#         if 'MASK' in mds:
#             mask = np.logical_or(mask, mds.MASK.values)
#         mds = mds.assign(**{
#                 'MASK': (('x', 'y'), da.from_array(mask, chunks=(-1, -1)))
#         })


#     model = da.from_array(model, chunks=(1, -1, -1))
#     mds = mds.assign(**{
#             'CLEAN_MODEL': (('band', 'x', 'y'), model)
#     })

#     dask.compute(xds_to_zarr(mds, mds_name, columns='ALL'))

#     if opts.do_residual:
#         print('Computing final residual', file=log)
#         # first write it to disk per dataset
#         out_ds = []
#         for ds in dds:
#             dirty = ds.DIRTY.data
#             wgt = ds.WEIGHT.data
#             uvw = ds.UVW.data
#             freq = ds.FREQ.data
#             beam = ds.BEAM.data
#             vis_mask = ds.MASK.data
#             b = ds.bandid
#             # we only want to apply the beam once here
#             residual = dirty - hessian(beam * model[b], uvw, wgt,
#                                        vis_mask, freq, None, hessopts)
#             ds = ds.assign(**{'CLEAN_RESIDUAL': (('x', 'y'), residual)})
#             out_ds.append(ds)

#         writes = xds_to_zarr(out_ds, dds_name, columns='CLEAN_RESIDUAL')
#         dask.compute(writes)

#     if opts.fits_mfs or opts.fits_cubes:
#         print("Writing fits files", file=log)
#         # construct a header from xds attrs
#         ra = dds[0].ra
#         dec = dds[0].dec
#         radec = [ra, dec]
#         cell_rad = dds[0].cell_rad
#         cell_deg = np.rad2deg(cell_rad)
#         freq_out = mds.freq.data
#         ref_freq = np.mean(freq_out)
#         hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

#         model_mfs = np.mean(model, axis=0)

#         save_fits(f'{basename}_clean_model_mfs.fits', model_mfs, hdr_mfs)

#         if opts.update_mask:
#             save_fits(f'{basename}_clean_mask.fits', mask, hdr_mfs)

#         if opts.do_residual:
#             dds = xds_from_zarr(dds_name, chunks={'band': 1})
#             residual = [da.zeros((nx, ny), chunks=(-1, -1)) for _ in range(nband)]
#             wsums = np.zeros(nband)
#             for ds in dds:
#                 b = ds.bandid
#                 wsums[b] += ds.WSUM.values[0]
#                 residual[b] += ds.CLEAN_RESIDUAL.data
#             wsum = np.sum(wsums)
#             residual = (da.stack(residual)/wsum).compute()

#             residual_mfs = np.sum(residual, axis=0)
#             save_fits(f'{basename}_clean_residual_mfs.fits',
#                       residual_mfs, hdr_mfs)

#         if opts.fits_cubes:
#             # need residual in Jy/beam
#             hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
#             save_fits(f'{basename}_clean_model.fits', model, hdr)

#             if opts.do_residual:
#                 fmask = wsums > 0
#                 residual[fmask] /= wsums[fmask, None, None]
#                 save_fits(f'{basename}_clean_residual.fits',
#                           residual, hdr)

#     print("All done here.", file=log)
