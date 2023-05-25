import numpy as np
from smoove.kanterp import kanterp
from finufft import nufft1d3
import matplotlib.pyplot as plt

def smooth_ant(amp, phase, w, xcoord, p, c,
               do_phase=True, niter=10, dof=2):
    idx = w > 0.0
    # we need at least two points to smooth
    if idx.sum() < 2:
        return np.ones(amp.size), np.zeros(phase.size), p, c
    amplin = np.interp(xcoord, xcoord[idx], amp[idx])
    _, samp, _ = kanterp(xcoord, amplin, w, niter=niter, nu=dof)
    if do_phase:
        phaselin = np.interp(xcoord, xcoord[idx], phase[idx])
        _, sphase, _ = kanterp(xcoord, phaselin, w/samp,
                               niter=niter, nu=dof)
    else:
        sphase = np.zeros(phase.size)
    return samp, sphase, p, c


def lthreshold(x, sigma, kind='l1'):
    if kind=='l0':
        return np.where(np.abs(x) > sigma, x, 0) * np.sign(x)
    elif kind=='l1':
        absx = np.abs(x)
        return np.where(absx > sigma, absx - sigma, 0) * np.sign(x)


def add_column(ms, col_name, like_col="DATA", like_type=None):
    if col_name not in ms.colnames():
        desc = ms.getcoldesc(like_col)
        desc['name'] = col_name
        desc['comment'] = desc['comment'].replace(" ", "_")  # got this from Cyril, not sure why
        dminfo = ms.getdminfo(like_col)
        dminfo["NAME"] =  "{}-{}".format(dminfo["NAME"], col_name)
        # if a different type is specified, insert that
        if like_type:
            desc['valueType'] = like_type
        ms.addcols(desc, dminfo)
    return ms

# not currently chunking over time
def accum_vis(data, flag, ant1, ant2, nant, ref_ant=-1):
    return da.blockwise(_accum_vis, 'afc',
                        data, 'rfc',
                        flag, 'rfc',
                        ant1, 'r',
                        ant2, 'r',
                        ref_ant, None,
                        new_axes={'a':nant},
                        dtype=np.complex128)


def _accum_vis(data, flag, ant1, ant2, ref_ant):
    return _accum_vis_impl(data[0], flag[0], ant1[0], ant2[0], ref_ant)

# @jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def _accum_vis_impl(data, flag, ant1, ant2, ref_ant):
    # select out reference antenna
    I = np.where((ant1==ref_ant) | (ant2==ref_ant))[0]
    data = data[I]
    flag = flag[I]
    ant1 = ant1[I]
    ant2 = ant2[I]

    # we can zero flagged data because they won't contribute to the FT
    data = np.where(flag, 0j, data)
    nrow, nchan, ncorr = data.shape
    nant = np.maximum(ant1.max(), ant2.max()) + 1
    if ref_ant == -1:
        ref_ant = nant-1
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
    vis[valid] = vis[valid]/counts[valid]
    return vis

def estimate_delay(vis_ant, freq, min_delay):
    return da.blockwise(_estimate_delay, 'ac',
                        vis_ant, 'afc',
                        freq, 'f',
                        min_delay, None,
                        dtype=np.float64)


def _estimate_delay(vis_ant, freq, min_delay):
    return _estimate_delay_impl(vis_ant[0], freq[0], min_delay)

def _estimate_delay_impl(vis_ant, freq, min_delay):
    delta_freq = 1.0/min_delay
    nchan = freq.size
    fmax = freq.min() + delta_freq
    fexcess = fmax - freq.max()
    freq_cell = freq[1]-freq[0]
    if fexcess > 0:
        npad = np.int(np.ceil(fexcess/freq_cell))
        npix = (nchan + npad)
    else:
        npix = nchan
    while npix%2:
        npix = good_size(npix+1)
    npad = npix - nchan
    lag = np.fft.fftfreq(npix, freq_cell)
    lag = Fs(lag)
    dlag = lag[1] - lag[0]
    nant, _, ncorr = vis_ant.shape
    delays = np.zeros((nant, ncorr), dtype=np.float64)
    for p in range(nant):
        for c in range(ncorr):
            vis_fft = np.fft.fft(vis_ant[p, :, c], npix)
            pspec = np.abs(Fs(vis_fft))
            if not pspec.any():
                continue
            delay_idx = np.argmax(pspec)
            # fm1 = lag[delay_idx-1]
            # f0 = lag[delay_idx]
            # fp1 = lag[delay_idx+1]
            # delays[p,c] = 0.5*dlag*(fp1 - fm1)/(fm1 - 2*f0 + fp1)
            # print(p, c, lag[delay_idx])
            delays[p,c] = lag[delay_idx]
    return delays


def _estimate_tec_impl(vis_ant, freq, tec_nyq, max_tec, fctr, srf=10):
    nuinv = fctr/freq
    npix = int(srf*max_tec/tec_nyq)
    print(npix)
    lag = np.linspace(-0.5*max_tec, 0.5*max_tec, npix)
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
            tecs[p,c] = lag[tec_idx]
            print(f'TEC for antenna {p} and correlation {c} = {lag[tec_idx]}')
    return tecs


def array2qcal_ds(gobj_amp, gobj_phase, time, freq, ant_names, fid, ddid, sid, fname):
    nant, ndir, ncorr = gobj_amp.shape
    ntime = time.size
    nchan = freq.size
    # gains not chunked on disk
    gain = np.zeros((ntime, nchan, nant, ndir, ncorr), dtype=np.complex128)
    for p in range(nant):
        for c in range(ncorr):
            gain[:, :, p, 0, c] = gobj_amp[p, 0, c](time, freq)*np.exp(1.0j*gobj_phase[p, 0, c](time, freq))
    gain = da.from_array(gain, chunks=(-1, -1, -1, -1, -1))
    gflags = da.zeros((ntime, nchan, nant, ndir), chunks=(-1, -1, -1, -1), dtype=np.int8)
    data_vars = {
        'gains':(('gain_time', 'gain_freq', 'antenna', 'direction', 'correlation'), gain),
        'gain_flags':(('gain_time', 'gain_freq', 'antenna', 'direction'), gflags)
    }
    gain_spec_tup = namedtuple('gains_spec_tup', 'tchunk fchunk achunk dchunk cchunk')
    attrs = {
        'DATA_DESC_ID': int(ddid),
        'FIELD_ID': int(fid),
        'FIELD_NAME': fname,
        'GAIN_AXES': ('gain_time', 'gain_freq', 'antenna', 'direction', 'correlation'),
        'GAIN_SPEC': gain_spec_tup(tchunk=(int(ntime),),
                                    fchunk=(int(nchan),),
                                    achunk=(int(nant),),
                                    dchunk=(int(1),),
                                    cchunk=(int(ncorr),)),
        'NAME': 'NET',
        'SCAN_NUMBER': int(sid),
        'TYPE': 'complex'
    }
    if ncorr==1:
        corrs = np.array(['XX'], dtype=object)
    elif ncorr==2:
        corrs = np.array(['XX', 'YY'], dtype=object)
    coords = {
        'gain_freq': (('gain_freq',), freq),
        'gain_time': (('gain_time',), time),
        'antenna': (('antenna'), ant_names),
        'correlation': (('correlation'), corrs),
        'direction': (('direction'), np.array([0], dtype=np.int32)),
        'f_chunk': (('f_chunk'), np.array([0], dtype=np.int32)),
        't_chunk': (('t_chunk'), np.array([0], dtype=np.int32))
    }
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def interp_gain_grid(gdct, ant_names):
    nant = ant_names.size
    ncorr, ntime, nfreq = gdct[ant_names[0]].shape
    time = gdct['time']
    assert time.size==ntime
    freq = gdct['frequencies']
    assert freq.size==nfreq

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
    '''
    Estimates size of data in GB where:

    nant    - number of antennas
    nhr     - length of observation in hours
    nsec    - integration time in seconds
    nchan   - number of channels
    ncorr   - number of correlations
    nbytes  - bytes per item (eg. 8 for complex64)
    '''
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
    wlow = (infreqs - delta_freq/2.0)/ref_freq
    whigh = (infreqs + delta_freq/2.0)/ref_freq
    wdiff = whigh - wlow

    # set design matrix for each component
    # look at Offringa and Smirnov 1706.06786
    Xfit = np.zeros([nband, order])
    for i in range(1, order+1):
        Xfit[:, i-1] = (whigh**i - wlow**i)/(i*wdiff)

    # we want to fit a function modeli = Xfit comps
    # where Xfit is the design matrix corresponding to an integrated
    # polynomial model. The normal equations tells us
    # comps = (Xfit.T wsums Xfit)**{-1} Xfit.T wsums modeli
    # (Xfit.T wsums Xfit) == hesscomps
    # Xfit.T wsums modeli == dirty_comps

    dirty_comps = Xfit.T.dot(wsums*beta)

    hess_comps = Xfit.T.dot(wsums*Xfit)

    comps = np.linalg.solve(hess_comps, dirty_comps)

    # now we want to evaluate the unintegrated polynomial coefficients
    # the corresponding design matrix is constructed for a polynomial of
    # the form
    # modeli = comps[0]*1 + comps[1] * w + comps[2] w**2 + ...
    # where w = outfreqs/ref_freq
    w = outfreqs/ref_freq
    # nchan = outfreqs
    # Xeval = np.zeros((nchan, order))
    # for c in range(nchan):
    #     Xeval[:, c] = w**c
    Xeval = np.tile(w, order)**np.arange(order)

    betaout = Xeval.dot(comps)

    modelout = np.zeros((nchan, nx, ny))
    modelout[:, mask] = betaout

    return modelout
