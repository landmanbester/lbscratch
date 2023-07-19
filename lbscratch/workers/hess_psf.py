from contextlib import ExitStack
from lbscratch.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('lbscratch')
log = pyscilog.get_logger('HESS_PSF')

import numpy as np
import dask
import dask.array as da
from daskms.experimental.zarr import xds_from_zarr
from pathlib import Path
from scipy.constants import c as lightspeed
from ducc0.wgridder.experimental import vis2dirty, dirty2vis
from ducc0.fft import good_size, r2c, c2r, c2c
iFs = np.fft.ifftshift
Fs = np.fft.fftshift
import matplotlib.pyplot as plt
from lbscratch.pygridder import ms2dirty_numba, ms2dirty_wplane, dirty2ms_python_fast
import time


from scabha.schema_utils import clickify_parameters
from lbscratch.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.hess_psf["inputs"].keys():
    defaults[key] = schema.hess_psf["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.hess_psf)
def hess_psf(**kw):
    '''
    Find best reference antenna based on flagged percentages
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # pyscilog.log_to_file(f'hess_psf_{timestamp}.log')
    OmegaConf.set_struct(opts, True)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    xds_path = Path(opts.xds).resolve()
    xds_name = str(xds_path)
    xds = xds_from_zarr(xds_name,
                        chunks={'row': -1, 'chan': -1})

    nrow, nchan = xds[0].VIS.values.shape
    I = slice(0, nrow, 50)
    vis = xds[0].VIS.values[I]
    wgt = xds[0].WEIGHT.values[I]
    wgt[...] = 1.0  # start simple
    uvw = xds[0].UVW.values[I]
    freq = xds[0].FREQ.values
    mask = xds[0].MASK.values[I]

    umax = np.abs(uvw[:, 0]).max()
    vmax = np.abs(uvw[:, 1]).max()
    uv_max = np.maximum(umax, vmax)

    # max cell size
    cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)
    srf = 10.0
    cell_rad = cell_N / srf
    cell_deg = np.rad2deg(cell_rad)
    fov = 0.5
    npix = int(fov / cell_deg)
    if npix%2:
        npix += 1
    npix = good_size(npix)
    nx = npix
    ny = npix

    # we need to compute this for the size of the PSF
    x, y = np.meshgrid(*[-ss/2 + np.arange(ss) for ss in (2*nx, 2*ny)], indexing='ij')
    x *= cell_rad
    y *= cell_rad
    eps = x**2+y**2
    nm1 = -eps/(np.sqrt(1.-eps)+1.)

    epsilon = 1e-7

    # PSF with wgridder no wstack
    psf = vis2dirty(uvw=uvw,
                    freq=freq,
                    vis=wgt.astype(vis.dtype),
                    npix_x=2*nx, npix_y=2*ny,
                    pixsize_x=cell_rad, pixsize_y=cell_rad,
                    center_x=0.0, center_y=0.0,
                    epsilon=epsilon,
                    flip_v=False,
                    do_wgridding=False,
                    divide_by_n=False,  # I believe we won't need this since applied during convolution?
                    nthreads=8,
                    verbosity=2)

    psfhat = c2c(iFs(psf, axes=(0,1)),
                 axes=(0,1), inorm=0, forward=True)

    x = np.zeros((nx,  ny))
    x[nx//2, ny//2] = 1
    x[nx//3, ny//3] = 1

    # is divide_by_n supposed to have an effect when do_wgridding=False?
    res1 = hessian(x/(nm1+1)[nx//2:3*nx//2, ny//2:3*ny//2],
                   uvw, freq, cell_rad,
                   wstack=False,
                   epsilon=epsilon,
                   divn=True,
                   nthreads=8)
    res1 /= (nm1+1)[nx//2:3*nx//2, ny//2:3*ny//2]

    res2 = psf_convolve_slice(x, psfhat, psf.shape[-1], nm1,
                              divn=True,
                              nthreads=8)

    rmax = 2*np.abs(res1).max()  # avoid div by zero
    assert np.allclose(rmax + res1, rmax + res2, rtol=epsilon)

    # for reference what is the magnitude of the diffirence with w-gridding
    psfw = vis2dirty(uvw=uvw,
                    freq=freq,
                    vis=wgt.astype(vis.dtype),
                    npix_x=2*nx, npix_y=2*ny,
                    pixsize_x=cell_rad, pixsize_y=cell_rad,
                    center_x=0.0, center_y=0.0,
                    epsilon=epsilon,
                    flip_v=False,
                    do_wgridding=True,
                    divide_by_n=False,  # I believe we won't need this since applied during convolution?
                    nthreads=8,
                    verbosity=2)

    psfwhat = r2c(iFs(psfw, axes=(0,1)),
                  axes=(0,1), inorm=0, forward=True)

    res1 = hessian(x, uvw, freq, cell_rad, True, epsilon, False, 8)
    res2 = psf_convolve_slice(x, psfwhat, psf.shape[-1], nm1, False, 8)
    res3 = psf_convolve_slice(x, psfhat, psf.shape[-1], nm1, False, 8)

    print('Max diff between hess and psf-convolve-slice (psf with w-stacking) =', np.abs(res1-res2).max()/np.abs(res1).max())
    print('Max diff between hess and psf-convolve-slice (psf without w-stacking) =', np.abs(res1-res3).max()/np.abs(res1).max())


    psf2 = ms2dirty_numba(uvw, freq, wgt, 2*nx, 2*ny, cell_rad, cell_rad, epsilon, True, False)
    try:
        assert np.allclose(rmax + psfw, rmax + psf2, rtol=epsilon)
    except:
        print('Max diff between psfs with w-stacking in numba vs wgridder =', np.abs(psfw-psf2).max()/np.abs(psfw).max())

    # this one constructs the PSF per w-stack and is consistent with the numba implementation
    # but now probably not 100% correct?
    psfw_stack, nm12, w0, dw = ms2dirty_wplane(uvw, freq, wgt, 2*nx, 2*ny, cell_rad, cell_rad, epsilon, divn=False)

    # assert np.allclose(rmax + psf2, rmax + np.sum(psfw_stack, axis=0), rtol=epsilon)
    assert np.allclose(1+nm1, 1+nm12, rtol=1e-14)

    psfw_stack = iFs(psfw_stack, axes=(1,2))
    psfwhat_stack = c2c(psfw_stack, axes=(1,2), inorm=0)
    lastsize = psfw.shape[-1]


    res1 = hessian_python(x, uvw, freq, cell_rad, wstack=True, epsilon=epsilon, divn=False)
    res2 = psf_convolve_wplanes(x, psfwhat_stack, lastsize, nm1, w0, dw, divn=False, nthreads=8)



    plt.figure(1)
    plt.imshow(res1)
    plt.colorbar()
    plt.figure(2)
    plt.imshow(res2)
    plt.colorbar()
    plt.figure(3)
    plt.imshow(res1-res2)
    plt.colorbar()
    # plt.figure(4)
    # plt.imshow(res1-res3)
    # plt.colorbar()
    # plt.figure(5)
    # plt.imshow(res2-res3)
    # plt.colorbar()
    plt.show()

    print(np.abs(res1-res2).max()/np.abs(res1).max())


def psf_convolve_slice(
                    x,
                    psfhat,
                    lastsize,
                    nm1,
                    divn=True,
                    nthreads=1):
    '''
    Should be consistent with hessian + no wgridding + divn
    '''
    nx, ny = x.shape
    padding = ((nx//2, nx//2), (ny//2, ny//2))
    xpad = np.pad(x, padding, mode='constant')
    if divn:
        xpad /= (nm1+1)
    xhat = r2c(xpad, axes=(0, 1), nthreads=nthreads,
                forward=True, inorm=0)
    xhat *= psfhat
    xout = c2r(xhat, axes=(0, 1), forward=False,
            lastsize=lastsize, inorm=2, nthreads=nthreads,
            allow_overwriting_input=True)[nx//2:3*nx//2, ny//2:3*ny//2]  #[0:nx, 0:ny]
    if divn:
        return xout/(nm1+1)[nx//2:3*nx//2, ny//2:3*ny//2]
    else:
        return xout


def psf_convolve_wplanes(
                    x,  # input image, not overwritten
                    psfhat,
                    lastsize,
                    nm1,
                    w0,
                    dw,
                    divn=True,
                    nthreads=1):
    '''
    Should be consistent with hessian + wgridding + divn
    '''
    nx, ny = x.shape
    nw, _, _ = psfhat.shape
    padding = ((nx//2, nx//2), (ny//2, ny//2))
    xpad = np.pad(x, padding, mode='constant')
    if divn:
        xpad /= (nm1+1)
    xout = np.zeros_like(xpad)
    for w in range(nw):
        # xhat = c2c(xpad*np.exp(2j*np.pi*nm1*(w0+w*dw)), axes=(0, 1), nthreads=nthreads,
        #         forward=True, inorm=0)
        xhat = c2c(xpad, axes=(0, 1), nthreads=nthreads,
                    forward=True, inorm=0)
        xout += (c2c(xhat * psfhat[w], axes=(0, 1), forward=False,
                    inorm=2, nthreads=nthreads)).real
        # xout += (c2c(xhat * psfhat[w], axes=(0, 1), forward=False,
        #             inorm=2, nthreads=nthreads) * np.exp(-2j*np.pi*nm1*(w0+w*dw))).real

    if divn:
        return (xout/(nm1+1))[nx//2:3*nx//2, ny//2:3*ny//2]
    else:
        return xout[nx//2:3*nx//2, ny//2:3*ny//2]


def hessian_python(x, uvw, freq, cell, wstack=True, epsilon=1, divn=True):
    nx, ny = x.shape
    vis = dirty2ms_python_fast(uvw, freq, x, cell, cell, epsilon, wstack, divn)

    return ms2dirty_numba(uvw, freq, vis, nx, ny, cell, cell,
                               epsilon, wstack, divn)


def hessian(x, uvw, freq, cell, wstack=True, epsilon=1e-7, divn=True, nthreads=1):
    nx, ny = x.shape
    vis = dirty2vis(uvw=uvw,
                    freq=freq,
                    dirty=x,
                    pixsize_x=cell,
                    pixsize_y=cell,
                    epsilon=epsilon,
                    nthreads=nthreads,
                    do_wgridding=wstack,
                    divide_by_n=divn)

    return vis2dirty(uvw=uvw,
                     freq=freq,
                     vis=vis,
                     npix_x=nx,
                     npix_y=ny,
                     pixsize_x=cell,
                     pixsize_y=cell,
                     epsilon=epsilon,
                     nthreads=nthreads,
                     do_wgridding=wstack,
                     divide_by_n=divn)
