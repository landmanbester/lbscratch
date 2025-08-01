from collections import namedtuple

import dask.array as da
from ducc0.fft import good_size
from finufft import nufft1d3
import matplotlib.pyplot as plt
from numba import jit, njit
from numba.types import literal
import numpy as np
from pfb.utils.misc import chunkify_rows
from scipy.interpolate import RectBivariateSpline as rbs
from smoove.kanterp import kanterp
import sympy as sm
from sympy.physics.quantum import TensorProduct
from sympy.utilities.lambdify import lambdify
import xarray as xr

DIAG_DIAG = 0
DIAG = 1
FULL = 2


def smooth_ant(amp, phase, w, xcoord, p, c, do_phase=True, niter=10, dof=2, sigman_min=1e-2):
    idx = w > 0.0
    # we need at least two points to smooth
    if idx.sum() < 2:
        return np.ones(amp.size), np.zeros(phase.size), p, c
    amplin = np.interp(xcoord, xcoord[idx], amp[idx])
    theta, samp, _, nu = kanterp(xcoord, amplin, w, niter=niter, nu=dof, sigman_min=sigman_min)
    # print('amp', p, c, theta, nu)
    if do_phase:
        phaselin = np.interp(xcoord, xcoord[idx], phase[idx])
        theta, sphase, _, nu = kanterp(
            xcoord, phaselin, w / samp, niter=niter, nu=dof, sigman_min=sigman_min
        )
        # print('phase', p, c, theta, nu)
    else:
        sphase = np.zeros(phase.size)
    return samp, sphase, p, c


def lthreshold(x, sigma, kind="l1"):
    if kind == "l0":
        return np.where(np.abs(x) > sigma, x, 0) * np.sign(x)
    elif kind == "l1":
        absx = np.abs(x)
        return np.where(absx > sigma, absx - sigma, 0) * np.sign(x)


def add_column(ms, col_name, like_col="DATA", like_type=None):
    if col_name not in ms.colnames():
        desc = ms.getcoldesc(like_col)
        desc["name"] = col_name
        desc["comment"] = desc["comment"].replace(" ", "_")  # got this from Cyril, not sure why
        dminfo = ms.getdminfo(like_col)
        dminfo["NAME"] = "{}-{}".format(dminfo["NAME"], col_name)
        # if a different type is specified, insert that
        if like_type:
            desc["valueType"] = like_type
        ms.addcols(desc, dminfo)
    return ms


# not currently chunking over time
def accum_vis(data, flag, ant1, ant2, nant, ref_ant=-1):
    return da.blockwise(
        _accum_vis,
        "afc",
        data,
        "rfc",
        flag,
        "rfc",
        ant1,
        "r",
        ant2,
        "r",
        ref_ant,
        None,
        new_axes={"a": nant},
        dtype=np.complex128,
    )


def _accum_vis(data, flag, ant1, ant2, ref_ant):
    return _accum_vis_impl(data[0], flag[0], ant1[0], ant2[0], ref_ant)


# @jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def _accum_vis_impl(data, flag, ant1, ant2, ref_ant):
    # select out reference antenna
    idx = np.where((ant1 == ref_ant) | (ant2 == ref_ant))[0]
    data = data[idx]
    flag = flag[idx]
    ant1 = ant1[idx]
    ant2 = ant2[idx]

    # we can zero flagged data because they won't contribute to the FT
    data = np.where(flag, 0j, data)
    nrow, nchan, ncorr = data.shape
    nant = np.maximum(ant1.max(), ant2.max()) + 1
    if ref_ant == -1:
        ref_ant = nant - 1
    ncorr = data.shape[-1]
    vis = np.zeros((nant, nchan, ncorr), dtype=np.complex128)
    counts = np.zeros((nant, nchan, ncorr), dtype=np.float64)
    for row in range(nrow):
        p = int(ant1[row])
        q = int(ant2[row])
        if p == ref_ant:
            vis[q] += data[row].astype(np.complex128).conj()
            counts[q] += flag[row].astype(np.float64)
        elif q == ref_ant:
            vis[p] += data[row].astype(np.complex128)
            counts[p] += flag[row].astype(np.float64)
    valid = counts > 0
    vis[valid] = vis[valid] / counts[valid]
    return vis


def estimate_delay(vis_ant, freq, min_delay):
    return da.blockwise(
        _estimate_delay, "ac", vis_ant, "afc", freq, "f", min_delay, None, dtype=np.float64
    )


def _estimate_delay(vis_ant, freq, min_delay):
    return _estimate_delay_impl(vis_ant[0], freq[0], min_delay)


def _estimate_delay_impl(vis_ant, freq, min_delay):
    delta_freq = 1.0 / min_delay
    nchan = freq.size
    fmax = freq.min() + delta_freq
    fexcess = fmax - freq.max()
    freq_cell = freq[1] - freq[0]
    if fexcess > 0:
        npad = int(np.ceil(fexcess / freq_cell))
        npix = nchan + npad
    else:
        npix = nchan
    while npix % 2:
        npix = good_size(npix + 1)
    npad = npix - nchan
    lag = np.fft.fftfreq(npix, freq_cell)
    lag = np.fft.fftshift(lag)
    nant, _, ncorr = vis_ant.shape
    delays = np.zeros((nant, ncorr), dtype=np.float64)
    for p in range(nant):
        for c in range(ncorr):
            vis_fft = np.fft.fft(vis_ant[p, :, c], npix)
            pspec = np.abs(np.fft.fftshift(vis_fft))
            if not pspec.any():
                continue
            delay_idx = np.argmax(pspec)
            # fm1 = lag[delay_idx-1]
            # f0 = lag[delay_idx]
            # fp1 = lag[delay_idx+1]
            # delays[p,c] = 0.5*dlag*(fp1 - fm1)/(fm1 - 2*f0 + fp1)
            # print(p, c, lag[delay_idx])
            delays[p, c] = lag[delay_idx]
    return delays


def _estimate_tec_impl(vis_ant, freq, tec_nyq, max_tec, fctr, srf=10):
    nuinv = fctr / freq
    npix = int(srf * max_tec / tec_nyq)
    print(npix)
    lag = np.linspace(-0.5 * max_tec, 0.5 * max_tec, npix)
    nant, _, ncorr = vis_ant.shape
    tecs = np.zeros((nant, ncorr), dtype=np.float64)
    for p in range(nant):
        for c in range(ncorr):
            vis_fft = nufft1d3(nuinv, vis_ant[p, :, c], lag, eps=1e-8, isign=-1)
            pspec = np.abs(vis_fft)
            plt.plot(lag, pspec)
            # plt.arrow(tecsin[p,c], 0, 0.0, pspec.max())
            plt.show()
            if not pspec.any():
                continue
            tec_idx = np.argmax(pspec)
            # fm1 = lag[delay_idx-1]
            # f0 = lag[delay_idx]
            # fp1 = lag[delay_idx+1]
            # delays[p,c] = 0.5*dlag*(fp1 - fm1)/(fm1 - 2*f0 + fp1)
            # print(p, c, lag[delay_idx])
            tecs[p, c] = lag[tec_idx]
            print(f"TEC for antenna {p} and correlation {c} = {lag[tec_idx]}")
    return tecs


def array2qcal_ds(gobj_amp, gobj_phase, time, freq, ant_names, fid, ddid, sid, fname):
    nant, ndir, ncorr = gobj_amp.shape
    ntime = time.size
    nchan = freq.size
    # gains not chunked on disk
    gain = np.zeros((ntime, nchan, nant, ndir, ncorr), dtype=np.complex128)
    for p in range(nant):
        for c in range(ncorr):
            gain[:, :, p, 0, c] = gobj_amp[p, 0, c](time, freq) * np.exp(
                1.0j * gobj_phase[p, 0, c](time, freq)
            )
    gain = da.from_array(gain, chunks=(-1, -1, -1, -1, -1))
    gflags = da.zeros((ntime, nchan, nant, ndir), chunks=(-1, -1, -1, -1), dtype=np.int8)
    data_vars = {
        "gains": (("gain_time", "gain_freq", "antenna", "direction", "correlation"), gain),
        "gain_flags": (("gain_time", "gain_freq", "antenna", "direction"), gflags),
    }
    gain_spec_tup = namedtuple("gain_spec_tup", "tchunk fchunk achunk dchunk cchunk")
    attrs = {
        "DATA_DESC_ID": int(ddid),
        "FIELD_ID": int(fid),
        "FIELD_NAME": fname,
        "GAIN_AXES": ("gain_time", "gain_freq", "antenna", "direction", "correlation"),
        "GAIN_SPEC": gain_spec_tup(
            tchunk=(int(ntime),),
            fchunk=(int(nchan),),
            achunk=(int(nant),),
            dchunk=(1,),
            cchunk=(int(ncorr),),
        ),
        "NAME": "NET",
        "SCAN_NUMBER": int(sid),
        "TYPE": "complex",
    }
    if ncorr == 1:
        corrs = np.array(["XX"], dtype=object)
    elif ncorr == 2:
        corrs = np.array(["XX", "YY"], dtype=object)
    coords = {
        "gain_freq": (("gain_freq",), freq),
        "gain_time": (("gain_time",), time),
        "antenna": (("antenna"), ant_names),
        "correlation": (("correlation"), corrs),
        "direction": (("direction"), np.array([0], dtype=np.int32)),
        "f_chunk": (("f_chunk"), np.array([0], dtype=np.int32)),
        "t_chunk": (("t_chunk"), np.array([0], dtype=np.int32)),
    }
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def interp_gain_grid(gdct, ant_names):
    nant = ant_names.size
    ncorr, ntime, nfreq = gdct[ant_names[0]].shape
    time = gdct["time"]
    assert time.size == ntime
    freq = gdct["frequencies"]
    assert freq.size == nfreq

    gain = np.zeros((ntime, nfreq, nant, 1, ncorr), dtype=np.complex128)

    # get axes in qcal order
    for p, name in enumerate(ant_names):
        gain[:, :, p, 0, :] = np.moveaxis(gdct[name], 0, -1)

    # fit spline to time and freq axes
    gobj_amp = np.zeros((nant, 1, ncorr), dtype=object)
    gobj_phase = np.zeros((nant, 1, ncorr), dtype=object)
    for p in range(nant):
        for c in range(ncorr):
            gobj_amp[p, 0, c] = rbs(time, freq, np.abs(gain[:, :, p, 0, c]))
            unwrapped_phase = np.unwrap(np.unwrap(np.angle(gain[:, :, p, 0, c]), axis=0), axis=1)
            gobj_phase[p, 0, c] = rbs(time, freq, unwrapped_phase)
    return gobj_amp, gobj_phase


def estimate_data_size(nant, nhr, nsec, nchan, ncorr, nbytes):
    """
    Estimates size of data in GB where:

    nant    - number of antennas
    nhr     - length of observation in hours
    nsec    - integration time in seconds
    nchan   - number of channels
    ncorr   - number of correlations
    nbytes  - bytes per item (eg. 8 for complex64)
    """
    nbl = nant * (nant - 1) // 2
    ntime = nhr * 3600 // nsec
    return nbl * ntime * nchan * ncorr * nbytes / 1e9


def interp_cube(model, wsums, infreqs, outfreqs, ref_freq, spectral_poly_order):
    nband, nx, ny = model
    mask = np.any(model, axis=0)
    # components excluding zeros
    beta = model[:, mask]
    if spectral_poly_order > infreqs.size:
        raise ValueError("spectral-poly-order can't be larger than nband")

    # we are given frequencies at bin centers, convert to bin edges
    delta_freq = infreqs[1] - infreqs[0]
    wlow = (infreqs - delta_freq / 2.0) / ref_freq
    whigh = (infreqs + delta_freq / 2.0) / ref_freq
    wdiff = whigh - wlow

    # set design matrix for each component
    # look at Offringa and Smirnov 1706.06786
    Xfit = np.zeros([nband, spectral_poly_order])
    for i in range(1, spectral_poly_order + 1):
        Xfit[:, i - 1] = (whigh**i - wlow**i) / (i * wdiff)

    # we want to fit a function modeli = Xfit comps
    # where Xfit is the design matrix corresponding to an integrated
    # polynomial model. The normal equations tells us
    # comps = (Xfit.T wsums Xfit)**{-1} Xfit.T wsums modeli
    # (Xfit.T wsums Xfit) == hesscomps
    # Xfit.T wsums modeli == dirty_comps

    dirty_comps = Xfit.T.dot(wsums * beta)

    hess_comps = Xfit.T.dot(wsums * Xfit)

    comps = np.linalg.solve(hess_comps, dirty_comps)

    # now we want to evaluate the unintegrated polynomial coefficients
    # the corresponding design matrix is constructed for a polynomial of
    # the form
    # modeli = comps[0]*1 + comps[1] * w + comps[2] w**2 + ...
    # where w = outfreqs/ref_freq
    w = outfreqs / ref_freq
    # nchan = outfreqs
    # Xeval = np.zeros((nchan, order))
    # for c in range(nchan):
    #     Xeval[:, c] = w**c
    Xeval = np.tile(w, spectral_poly_order) ** np.arange(spectral_poly_order)

    betaout = Xeval.dot(comps)

    modelout = np.zeros((outfreqs.size, nx, ny))
    modelout[:, mask] = betaout

    return modelout


def stokes_vis(product, mode, pol="linear"):
    # symbolic variables
    gp00, gp10, gp01, gp11 = sm.symbols("g_{p00} g_{p10} g_{p01} g_{p11}", real=False)
    gq00, gq10, gq01, gq11 = sm.symbols("g_{q00} g_{q10} g_{q01} g_{q11}", real=False)
    w0, w1, w2, w3 = sm.symbols("W_0 W_1 W_2 W_3", real=True)
    v0, v1, v2, v3 = sm.symbols("v_{00} v_{10} v_{01} v_{11}", real=False)
    i, q, u, v = sm.symbols("I Q U V", real=True)

    # Jones matrices
    Gp = sm.Matrix([[gp00, gp01], [gp10, gp11]])
    Gq = sm.Matrix([[gq00, gq01], [gq10, gq11]])

    # Mueller matrix
    Mpq = TensorProduct(Gp, Gq.conjugate())

    # inverse noise covariance (weights)
    W = sm.Matrix([[w0, 0, 0, 0], [0, w1, 0, 0], [0, 0, w2, 0], [0, 0, 0, w3]])

    # visibilities
    Vpq = sm.Matrix([[v0], [v1], [v2], [v3]])

    # Stokes to corr operator
    if True:  # pol == 'linear':
        T = sm.Matrix([[1.0, 1.0, 0, 0], [0, 0, 1.0, -1.0j], [0, 0, 1.0, 1.0j], [1, -1, 0, 0]])
    else:
        raise NotImplementedError("Sorry, no circular pol")

    # corr weights
    M = T.H * Mpq.H * W * Mpq * T
    V = T.H * Mpq.H * W * Vpq

    wparams = (gp00, gp10, gp01, gp11, gq00, gq10, gq01, gq11, w0, w1, w2, w3)
    vparams = (gp00, gp10, gp01, gp11, gq00, gq10, gq01, gq11, w0, w1, w2, w3, v0, v1, v2, v3)
    if product == literal("I"):
        wfunc = lambdify(wparams, M[0, 0], "numpy")
        vfunc = lambdify(vparams, V[0], "numpy")
        # wfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(wparams, M[0, 0], 'numpy'))
        # vfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(vparams, V[0], 'numpy'))

    elif product == literal("Q"):
        wfunc = lambdify(wparams, M[1, 1], "numpy")
        vfunc = lambdify(vparams, V[1], "numpy")
        # wfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(wparams, M[1, 1], 'numpy'))
        # vfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(vparams, V[1], 'numpy'))

    elif product == literal("U"):
        wfunc = lambdify(wparams, M[1, 1], "numpy")
        vfunc = lambdify(vparams, V[1], "numpy")
        # wfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(wparams, M[2, 2], 'numpy'))
        # vfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(vparams, V[2], 'numpy'))

    elif product == literal("V"):
        wfunc = lambdify(wparams, M[1, 1], "numpy")
        vfunc = lambdify(vparams, V[1], "numpy")
        # wfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(wparams, M[3, 3], 'numpy'))
        # vfunc = njit(nogil=True, cache=True, fastmath=True)(lambdify(vparams, V[3], 'numpy'))
    else:
        raise ValueError(f"Unknown product {product}")

    if mode == DIAG_DIAG:
        # M = M.subs(gp01, 0)
        # M = M.subs(gp10, 0)
        # M = M.subs(gq01, 0)
        # M = M.subs(gq10, 0)
        # V = V.subs(gp01, 0)
        # V = V.subs(gp10, 0)
        # V = V.subs(gq01, 0)
        # V = V.subs(gq10, 0)
        # V = V.subs(v1, 0)
        # V = V.subs(v2, 0)
        # wparams = (gp00, gp11, gq00, gq11, w0, w3)
        # vparams = (gp00, gp11, gq00, gq11, w0, w3, v0, v3)
        # if product == literal('I'):
        #     wfunc = lambdify(wparams, M[0, 0], 'numpy')
        #     vfunc = lambdify(vparams, V[0], 'numpy')

        # elif product == 'Q':
        #     wfunc = lambdify(wparams, M[1, 1], 'numpy')
        #     vfunc = lambdify(vparams, V[1], 'numpy')

        # elif product == 'U':
        #     wfunc = lambdify(wparams, M[2, 2], 'numpy')
        #     vfunc = lambdify(vparams, V[2], 'numpy')

        # elif product == 'V':
        #     wfunc = lambdify(wparams, M[3, 3], 'numpy')
        #     vfunc = lambdify(vparams, V[3], 'numpy')
        # else:
        #     raise ValueError(f'Unknown product {product}')

        # @jit(nogil=True, cache=True, fastmath=True)
        def wgt_func(wgt, gp, gq):
            return wfunc(gp[0], 0j, 0j, gp[1], gq[0], 0j, 0j, gq[1], wgt[0], 0, 0, wgt[1]).real

        # @jit(nogil=True, cache=True, fastmath=True)
        def vis_func(data, wgt, gp, gq):
            return vfunc(
                gp[0],
                0j,
                0j,
                gp[1],
                gq[0],
                0j,
                0j,
                gq[1],
                wgt[0],
                0,
                0,
                wgt[1],
                data[0],
                0j,
                0j,
                data[1],
            )

    elif mode == DIAG:
        # M = M.subs(gp01, 0)
        # M = M.subs(gp10, 0)
        # M = M.subs(gq01, 0)
        # M = M.subs(gq10, 0)
        # V = V.subs(gp01, 0)
        # V = V.subs(gp10, 0)
        # V = V.subs(gq01, 0)
        # V = V.subs(gq10, 0)
        # wparams = (gp00, gp11, gq00, gq11, w0, w1, w2, w3)
        # vparams = (gp00, gp11, gq00, gq11, w0, w1, w2, w3, v0, v1, v2, v3)
        # if product == literal('I'):
        #     wfunc = lambdify(wparams, M[0, 0], 'numpy')
        #     vfunc = lambdify(vparams, V[0], 'numpy')

        # elif product == 'Q':
        #     wfunc = lambdify(wparams, M[1, 1], 'numpy')
        #     vfunc = lambdify(vparams, V[1], 'numpy')

        # elif product == 'U':
        #     wfunc = lambdify(wparams, M[2, 2], 'numpy')
        #     vfunc = lambdify(vparams, V[2], 'numpy')

        # elif product == 'V':
        #     wfunc = lambdify(wparams, M[3, 3], 'numpy')
        #     vfunc = lambdify(vparams, V[3], 'numpy')
        # else:
        #     raise ValueError(f'Unknown product {product}')

        @jit(nogil=True, cache=True, fastmath=True)
        def wgt_func(wgt, gp, gq):
            return wfunc(
                gp[0], 0j, 0j, gp[1], gq[0], 0j, 0j, gq[1], wgt[0], wgt[1], wgt[2], wgt[3]
            ).real

        @jit(nogil=True, cache=True, fastmath=True)
        def vis_func(data, wgt, gp, gq):
            return vfunc(
                gp[0],
                0j,
                0j,
                gp[1],
                gq[0],
                0j,
                0j,
                gq[1],
                wgt[0],
                wgt[1],
                wgt[2],
                wgt[3],
                data[0],
                data[1],
                data[2],
                data[3],
            )

    elif mode == FULL:
        # wparams = (gp00, gp10, gp01, gp11, gq00, gq10, gq01, gq11, w0, w1, w2, w3)
        # vparams = (gp00, gp10, gp01, gp11, gq00, gq10, gq01, gq11, w0, w1, w2, w3, v0, v1, v2, v3)
        # if product == 'I':
        #     wfunc = lambdify(wparams, M[0, 0], 'numpy')
        #     vfunc = lambdify(vparams, V[0], 'numpy')

        # elif product == 'Q':
        #     wfunc = lambdify(wparams, M[1, 1], 'numpy')
        #     vfunc = lambdify(vparams, V[1], 'numpy')

        # elif product == 'U':
        #     wfunc = lambdify(wparams, M[2, 2], 'numpy')
        #     vfunc = lambdify(vparams, V[2], 'numpy')

        # elif product == 'V':
        #     wfunc = lambdify(wparams, M[3, 3], 'numpy')
        #     vfunc = lambdify(vparams, V[3], 'numpy')
        # else:
        #     raise ValueError(f'Unknown product {product}')

        @jit(nogil=True, cache=True, fastmath=True)
        def wgt_func(wgt, gp, gq):
            return wfunc(
                gp[0, 0],
                gp[1, 0],
                gp[0, 1],
                gp[1, 1],
                gq[0, 0],
                gq[1, 0],
                gq[0, 1],
                gq[1, 1],
                wgt[0],
                wgt[1],
                wgt[2],
                wgt[3],
            ).real

        @jit(nogil=True, cache=True, fastmath=True)
        def vis_func(data, wgt, gp, gq):
            return vfunc(
                gp[0, 0],
                gp[1, 0],
                gp[0, 1],
                gp[1, 1],
                gq[0, 0],
                gq[1, 0],
                gq[0, 1],
                gq[1, 1],
                wgt[0],
                wgt[1],
                wgt[2],
                wgt[3],
                data[0],
                data[1],
                data[2],
                data[3],
            )

    return vfunc, wfunc


def row_to_tbl(data, ant1, ant2, time):
    nant = np.maximum(ant1, ant2) + 1
    nbl = nant * (nant - 1) / 2 + nant  # including autos
    utime = np.unique(time)
    ntime = utime.size
    dims = data.shape[1:]
    out_data = np.zeros((ntime, nbl, *dims), dtype=data.dtype)
    P, Q = np.tril_indices(nant)

    row_chunks, rbin_idx, rbin_counts = chunkify_rows(time, 1)

    for t in range(ntime):
        for row in range(rbin_idx[t], rbin_idx[t] + rbin_counts[t]):
            p = ant1[row]
            q = ant2[row]
            b = index_where(p, q, P, Q)
            out_data[t, b] = data[row]
    return out_data


def flautos(data, flag, ant1, ant2, time, sigma=3.0):
    row_counts, tbin_idx, tbin_counts = chunkify_rows(time, utimes_per_chunk=-1)
    return da.blockwise(
        _flautos,
        "rfc",
        data,
        "rfc",
        flag,
        "rfc",
        ant1,
        "r",
        ant2,
        "r",
        tbin_idx,
        None,
        tbin_counts,
        None,
        sigma,
        None,
        dtype=flag.dtype,
        meta=np.empty(flag.shape, dtype=flag.dtype),
    )


def _flautos(data, flag, ant1, ant2, tbin_idx, tbin_counts, sigma=3):
    # compute mad flags per antenna over time axis
    ntime = tbin_idx.size
    nrow, nchan, ncorr = data.shape
    nant = np.maximum(ant1.max(), ant2.max()) + 1
    ant_flags = np.zeros((nant, ntime, nchan, ncorr), dtype=bool)
    flag_ant(data, flag, ant_flags, ant1, ant2, sigma=sigma)

    # propagate to baselines i.e. if either antenna is flagged then flag baseline
    flag_row(data, flag, ant_flags, ant1, ant2, tbin_idx, tbin_counts)
    return flag


@njit
def max_abs_deviation(data):
    med = np.median(data)
    abs_med_deviation = np.abs(data - med)
    return med, 1.4826 * np.median(abs_med_deviation)


@njit
def flag_ant(data, flag, ant_flags, ant1, ant2, sigma=3.0, minabs=1e-3):
    nant, ntime, nchan, ncorr = ant_flags.shape
    for p in range(nant):
        # select autocorr
        idx = (ant1 == p) & (ant2 == p)
        datap = data[idx].real
        flagp = flag[idx]
        for f in range(nchan):
            datapf = datap[:, f]
            flagpf = flagp[:, f]
            for c in range(ncorr):
                datapfc = datapf[:, c]
                mask = ~flagpf[:, c]
                if not mask.any():
                    # data already flagged
                    continue
                d = np.abs(datapfc[mask])
                med, mad = max_abs_deviation(d)  # TODO adapt for abs
                min_val = np.maximum(minabs, med - sigma * mad)
                max_val = med + sigma * mad
                for t in range(ntime):
                    lowval = datapfc[t] <= min_val
                    highval = datapfc[t] >= max_val
                    ant_flags[p, t, f, c] = lowval or highval


@njit
def flag_row(data, flag, ant_flags, ant1, ant2, tbin_idx, tbin_counts):
    ntime = tbin_idx.size
    nrow, nchan, ncorr = data.shape
    for t in range(ntime):
        for r in range(tbin_idx[t], tbin_idx[t] + tbin_counts[t]):
            p = ant1[r]
            q = ant2[r]
            for f in range(nchan):
                for c in range(ncorr):
                    if ant_flags[p, t, f, c] or ant_flags[q, t, f, c]:
                        flag[r, f, c] = True


def set_flags(flag, idx):
    flag[:, idx, :] = True
    return flag


def flags_at_edges(flag, time):
    utime = np.unique(time)[((0, -1),)]  # only need first and last element
    nbl = (time == utime[0]).sum()
    edge_flags = da.blockwise(
        _flags_at_edges,
        "2rfc",
        flag,
        "rfc",
        time,
        "r",
        utime,
        None,
        new_axes={"2": 2},
        adjust_chunks={"row": nbl},
        dtype=np.uint8,
    )
    return edge_flags, utime


def _flags_at_edges(flag, time, utime):
    nrow, nchan, ncorr = flag.shape

    idxi = time == utime[0]
    idxf = time == utime[-1]

    nbl = idxi.sum()
    edge_flag = np.zeros((2, nbl, nchan, ncorr), dtype=np.uint8)

    edge_flag[0] = flag[idxi].astype(np.uint8)
    edge_flag[1] = flag[idxf].astype(np.uint8)
    return edge_flag


def index_where(p, q, P, Q):
    return ((P == p) & (Q == q)).argmax()


# Duplicate function removed - already defined above


def make_flag_row(flag):
    red_flag = da.blockwise(_red_flag, "rf", flag, "rfc", dtype=flag.dtype, adjust_chunks={"f": 1})

    return da.all(red_flag, axis=-1)


def _red_flag(flag):
    # first over corr axis
    f = np.all(flag[0], axis=-1)
    # now over chan
    return np.all(f, axis=-1, keepdims=True)
