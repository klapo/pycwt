import pycwt
from pycwt.helpers import rednoise
import numpy as np
from pycwt.helpers import find
from tqdm import tqdm
from scipy.signal import butter, filtfilt


def standardize(
        s,
    standardize=True,
    detrend=False,
    remove_mean=False,
    bandpass_filter=False,
    bandpass_kwargs=None,
):
    """
    Helper function for pre-processing data prior to fitting the CWT.

    Parameters
    ----------
        s : numpy array of shape (n,) to be normalized
        standardize : divide by the standard deviation
        detrend : Linearly detrend s, default is False. Only one of detrend and
            bandpass_filter may be True.
        remove_mean : remove the mean of s. Only one of detrend, remove_mean, and
            bandpass_filter may be True.
        bandpass_filter : band pass the data. Only one of detrend and bandpass_filter
            may be True.
            NOTE: At the moment I just implement a high-pass filter. This needs to
            be updated in the future to be a band pass instead.
        bandpass_kwargs : Dictionary of kwargs passed to the bandpass filter. Must
            be provided if bandpass_filter = True.
            Arguments:
                cutoff : float, desired cutoff frequency of the filter in Hz
                fs : float, sampling frequency in Hz
                order : int, order of the filter. Default is 5

    Returns
    ----------
        snorm : numpy array of shape (n,)
    """

    # @ Turn this into a decorator in the future for extra sexy python code
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="high", analog=False)
        return b, a

    def butter_highpass_filter(data, cutoff, fs, order=5):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    # Logic checking
    if detrend and remove_mean:
        raise ValueError(
            "Only standardize by either removing secular trend or mean, not both."
        )
    if (detrend and bandpass_filter) or (remove_mean and bandpass_filter):
        raise ValueError(
            "Standardizing can only take one of the following as True:"
            " remove_mean, detrend, bandpass_filer."
        )
    if bandpass_filter and bandpass_kwargs is None:
        raise ValueError(
            "When using the bandpass filter the bandpass_kwargs must be supplied."
        )

    # Derive the variance prior to any treatment
    smean = s.mean()

    # Remove the trend if requested
    if detrend:
        arbitrary_x = np.arange(0, s.size)
        p = np.polyfit(arbitrary_x, s, 1)
        snorm = s - np.polyval(p, arbitrary_x)
    else:
        snorm = s

    if remove_mean:
        snorm = snorm - smean

    if bandpass_filter:
        bandpass_kwargs.setdefault("order", 5)
        cutoff = bandpass_kwargs["cutoff"]
        order = bandpass_kwargs["order"]
        fs = bandpass_kwargs["fs"]
        snorm = butter_highpass_filter(s, cutoff, fs, order=order)

    # Standardize by the variance only after the above
    # data treatment.
    std = snorm.std()
    if standardize:
        snorm = snorm / std

    return snorm


def coi_scale_avg(coi, scales):
    """
    Returns the upper and lower coi bounds for scale averaged wavelet power.

    Parameters
    ----------
        coi - np.array (or similar)
            location  of the coi in period space
        scales - np.array (or similar)
            The cwt scales that label the coi

    Returns
    ----------
        min_index : list
            indices of the coi scales for the shortest scale in
            the scale averaging interval
        max_index : list
            indices of the coi scales for the shortest scale in
            the scale averaging interval
    """
    mindex1 = np.argmin(np.abs(coi[: len(coi) // 2] - np.min(scales)))
    mindex2 = np.argmin(np.abs(coi[len(coi) // 2 :] - np.min(scales)))
    min_index = [mindex1, mindex2 + len(coi) // 2]

    maxdex1 = np.argmin(np.abs(coi[: len(coi) // 2] - np.max(scales)))
    maxdex2 = np.argmin(np.abs(coi[len(coi) // 2 :] - np.max(scales)))
    max_index = [maxdex1, maxdex2 + len(coi) // 2]

    return min_index, max_index


class pyCWavelet:
    def __init__(self, signal_norm=None, sig_level=None, rectify=True):
        self._signal_norm = signal_norm
        self._dx = None
        self._x = None
        self._mother = None
        self._octaves = None
        self._scales_to_avg = None
        self._glbl_power_var_scaling = None
        self._norm_kwargs = None
        self._variance = None
        self._rectify = rectify
        self._significance_test = None
        self._sig_kwargs = None
        self._sig_level = sig_level
        self._global_power = None
        self._power = None
        self._fft_power = None
        self._period = None
        self._scales = None
        self._significance = None
        self._global_significance = None
        self._scale_avg = None
        self._scale_avg_significance = None

    def fit(
        self,
        signal,
        dx,
        x,
        mother,
        octaves=None,
        scales_to_avg=None,
        glbl_power_var_scaling=True,
        norm_kwargs=None,
        variance=None,
        rectify=True,
        significance_test=True,
        sig_kwargs=None,
        sig_level=None
    ):
        """Calculate the wavelet power.

        Note: this function is based on the example from the pycwt library
        (https://pycwt.readthedocs.io/en/latest/tutorial.html)

        The resulting spectra are rectified following Liu 2007

        Parameters
        ----------
            signal : ndarray
            dx : float
            x : ndarray
            mother : pycwt wavelet object
            octaves : tuple, optional
            scales_to_avg : list, optional
            glbl_power_var_scaling : boolean, optional
            norm_kwargs : dict
            variance : float, optional
            rectify : boolean, optional
            significance_test : boolean, optional
            sig_kwargs : dict, optional

        Returns
        ----------
        """

        # Unpack the inputs for fitting.
        self._dx = dx
        self._x = x
        self._mother = mother
        self._octaves = octaves
        self._scales_to_avg = scales_to_avg
        self._glbl_power_var_scaling = glbl_power_var_scaling
        self._rectify = rectify
        self._significance_test = significance_test
        self._sig_kwargs = sig_kwargs
        self._sig_level = sig_level

        if len(signal.shape) > 1:
            raise ValueError("The input signal should be a 1d object.")

        if self._norm_kwargs is None:
            self._norm_kwargs = {"detrend": True, "standardize": True}
        else:
            self._norm_kwargs.setdefault("standardize", True)
            self._norm_kwargs.setdefault("detrend", True)

        # If we do not normalize by the std, we need to find the variance
        if not self._norm_kwargs["standardize"] and self._variance is None:
            self._variance = np.std(signal) ** 2

        # For strongly non-stationary vectors, estimating the (white noise)
        # variance from the data itself is poorly defined. This option allows
        # the user to pass a pre-determined variance.
        elif self._variance is not None:
            self._variance = variance

        # If the data were standardized, the variance should be 1 by
        # definition (assuming normally distributed processes)
        else:
            self._variance = 1.0

        # Specify how the significance test should be returned.
        if self._significance_test and "local_sig_type" in self._sig_kwargs:
            self._local_sig_type = self._sig_kwargs["local_sig_type"]
        else:
            self._local_sig_type = "index"

        # Wavelet properties
        # Starting scale is twice the resolution
        s0 = 2 * self._dx
        # Default wavelet values yields 85 periods
        if self._octaves is None:
            # Twelve sub-octaves per octaves
            dj = 1 / 12
            # Seven powers of two with dj sub-octaves
            J = 7 / dj
        else:
            # Number of sub-octaves per octaves
            dj = self._octaves[0]
            # Number of powers of two with dj sub-octaves
            J = self._octaves[1]

        # Pre-process the input signal
        self._signal_norm = standardize(signal, **self._norm_kwargs)
        N = self._signal_norm.size

        # Perform the wavelet transform using the parameters defined above.
        wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(
            self._signal_norm, self._dx, dj, s0, J, self._mother
        )

        # Calculate the normalized wavelet and Fourier power spectra,
        # as well as the Fourier equivalent periods for each wavelet scale.
        # Note that this power is not normalized as in TC98 equation 8,
        # the factor of dt^(1/2) is missing.
        self._power = np.abs(wave) ** 2
        # fft_power = np.abs(fft) ** 2
        self._period = 1 / freqs
        self._scales = scales

        # Do the significance testing for both the local and global spectra
        if self._significance_test:
            if "alpha" not in self._sig_kwargs:
                # Lag-1 auto-correlation for red noise-based significance
                # testing.
                alpha, _, _ = pycwt.ar1(self._signal_norm)
            else:
                alpha = self._sig_kwargs["alpha"]

            # Local power spectra significance test, where the ratio power /
            # signif > 1.
            signif, _ = pycwt.significance(
                self._variance,
                self._dx,
                self._scales,
                0,
                alpha,
                significance_level=self._sig_level,
                wavelet=self._mother,
            )
            self._significance = np.ones([1, N]) * signif[:, None]
            # The default behavior is to return a matrix that can be directly
            # used to draw a single contour following the pycwt_plot_helper
            # functions. The other option is to just return the local
            # significance levels as a matrix with dimensions equal to 'wave'.
            if self._local_sig_type == "index":
                self._significance = self._power / self._significance

        # Rectify the power spectrum according to Liu et al. (2007)[2]
        if self._rectify:
            self._power /= self._scales[:, None]

        # Calculate the global wavelet spectrum
        self._global_power = self._power.mean(axis=1)
        if self._glbl_power_var_scaling:
            self._global_power = self._global_power / self._variance

        # Do the significance testing for the global spectra
        if self._significance_test:
            # Global power spectra significance test. Note: this variable  is
            # different than the local significance variable. Here the global
            # spectra is reported directly.
            dof = N - self._scales  # Correction for padding at edges
            self._global_significance, _ = pycwt.significance(
                self._variance,
                self._dx,
                self._scales,
                1,
                alpha,
                significance_level=self._sig_level,
                dof=dof,
                wavelet=self._mother,
            )
        # Just return nans in the significance variables as they won't break
        # e.g. plotting routines.
        else:
            self._global_significance = np.ones_like(self._global_power) * np.nan
            self._significance = np.ones_like(self._power) * np.nan

        if self._rectify:
            self._global_significance = self._global_significance / self._scales

        if self._scales_to_avg is not None:
            Cdelta = self._mother.cdelta
            num_intervals = np.shape(scales_to_avg)[0]
            self._scale_avg = np.ones((len(self._x), num_intervals))
            self._scale_avg_significance = np.ones(num_intervals)

            for nint in np.arange(num_intervals):
                # Calculate the scale average between two periods and the
                # significance level.
                sel = find(
                    (self._period >= self._scales_to_avg[nint][0])
                    & (self._period < self._scales_to_avg[nint][1])
                )

                # In the pycwt example and TC98 eq. 24 the scale-avg power is
                # scale rectified. However, the rectification occurs above in
                # the calculation of power. Do not repeat here.
                if self._rectify:
                    self._scale_avg[:, nint] = (
                        dj * dx / Cdelta * self._power[sel, :].sum(axis=0)
                    )
                # If no rectification is requested, then for consistency with
                # TC98 we return the rectified power here.
                elif not self._rectify:
                    # The example code is just included here as a reference
                    # scale_avg = (scales * np.ones((N, 1))).transpose()
                    # scale_avg = power / scale_avg
                    # As in Torrence and  Compo (1998) equation 24 TC98 eq.
                    # 24 does not include the variance as in the tutorial.
                    self._scale_avg[:, nint] = (
                        dj
                        * self._dx
                        / Cdelta
                        * self._power[sel, :].sum(axis=0)
                        / self._scales[:, None]
                    )

                if self._significance_test:
                    try:
                        self._scale_avg_significance[nint], _ = pycwt.significance(
                            self._variance,
                            self._dx,
                            self._scales,
                            2,
                            alpha,
                            significance_level=self._sig_level,
                            dof=[scales[sel[0]], scales[sel[-1]]],
                            wavelet=self._mother,
                        )
                    except IndexError:
                        # One of the scales to average was outside the range
                        # of the CWT's scales. Return no significance level
                        # for this averaging interval and move to the next one.
                        self._scale_avg_significance[nint] = np.nan
                        continue
                else:
                    self._scale_avg_significance[nint] = np.nan

        # @ToDo: Rename period and scales to better reflect that period is the
        # inverse fourier frequencies and that scales are the inverse wavelet
        # frequencies.
        return self

    def wavelet_coherent(
        self,
        s1,
        s2,
        dx,
        dj,
        s0,
        J,
        mother,
        norm_kwargs=None,
    ):
        """
        Calculate coherence between two vectors for a given wavelet. Currently
        only the Morlet wavelet is implemented. Returns the wavelet linear coherence
        (also known as coherency and wavelet coherency transform), the cross-wavelet
        transform, and the phase angle between signals.

        Parameters
        ----------
            s1 : 1D numpy array or similar object of length N.
                Signal one to perform the wavelet linear coherence with s2. Assumed
                to be pre-formatted by the `standardaize` function. s1 and s2 must
                be the same size.
            s2 : 1D numpy array or similar object of length N.
                Signal two to perform the wavelet linear coherence with s1. Assumed
                to be pre-formatted by the `standardaize` function. s1 and s2 must
                be the same size.
            dx : scalar or float. Spacing between elements in s1 and s2. s1 and s2
                are assumed to have equal spacing.
            dj : float, number of suboctaves per octave expressed as a fraction.
                Default value is 1 / 12 (12 suboctaves per octave)
            s0 : Smallest scale of the wavelet (if unsure = 2 * dt)
            J : float, number of octaves expressed as a power of 2 (e.g., the
                default value of 7 / dj means 7 powers of 2 octaves.
            mother : pycwt wavelet object, 'Morlet' is the only valid selection as
                a result of requiring an analytic expression for smoothing the
                wavelet.
            norm_kwargs : dict
                Keywords to pass to `standardize`. Default arguments are
                    {"detrend": True, "standardize": True}

        Returns
        ----------
            WCT : same type as s1 and s2, size PxN
                The biwavelet linear coherence transform.
            aWCT : same type as s1 and s2, size of PxN
                phase angle between s1 and s2.
            W12 : same type as s1 and s2, size PxN
                The cross-wavelet transform power, unrectified.
            W12_corr : the same type as s1 and s2, size PxN
                Cross-wavelet power, rectified following Veleda et al., 2012 and
                the R biwavelet package.
            period : numpy array, length P
                Fourier mode inverse frequencies.
            coi : numpy array  of length n, cone of influence
            angle : phase angle in degrees
            w1 : same type as s1, size PxN
                CWT for s1
            w2 : same type as s1, size PxN
                CWT for s2
        """
        assert (
            mother.name == "Morlet"
        ), "XWT requires smoothing, which is only available to the Morlet mother."
        wavelet_obj = pycwt.wavelet._check_parameter_wavelet(mother)

        assert np.size(s1) == np.size(s2), "s1 and s2 must be the same size."

        # s1 and s2 MUST be the same size
        assert s1.shape == s2.shape, "Input signals must share the exact same shape."

        if norm_kwargs is None:
            self._norm_kwargs = {"detrend": True, "standardize": True}
        else:
            self._norm_kwargs = norm_kwargs

        s1_norm = standardize(s1, **self._norm_kwargs)
        s2_norm = standardize(s2, **self._norm_kwargs)

        # Calculates the CWT of the time-series making sure the same parameters
        # are used in both calculations.
        W1, sj, freq, coi, _, _ = pycwt.cwt(s1_norm, dx, dj, s0, J, mother)
        W2, _, _, _, _, _ = pycwt.cwt(s2_norm, dx, dj, s0, J, mother)

        # We need a 2D matrix for the math that follows
        scales = np.atleast_2d(sj)
        periods = np.atleast_2d(1 / freq)

        # Perform the cross-wavelet transform
        W12 = W1 * W2.conj()
        # Here I follow the R biwavelet package for the implementation of the
        # scale rectification. Note that the R package normalizes by the largest
        # wavelet scale. I choose to not include that scaling factor here.
        # W12_corr = W1 * W2.conj() * np.max(periods) / periods.T
        W12_corr = W1 * W2.conj() / periods.T

        # Coherence

        # Smooth the wavelet spectra before truncating.
        if mother.name == "Morlet":
            sW1 = wavelet_obj.smooth((np.abs(W1) ** 2 / scales.T), dx, dj, sj)
            sW2 = wavelet_obj.smooth((np.abs(W2) ** 2 / scales.T), dx, dj, sj)
            sW12 = wavelet_obj.smooth((W12 / scales.T), dx, dj, sj)
        WCT = np.abs(sW12) ** 2 / (sW1 * sW2)
        aWCT = np.angle(W12)
        # @ fix this incorrect angle conversion.
        angle = (0.5 * np.pi - aWCT) * 180 / np.pi

        # @ better names to reflect fourier vs wavelet frequency/scale
        scales = np.squeeze(scales)

        return WCT, aWCT, W12, W12_corr, 1 / freq, coi, angle, sW1, sW2

    def wct_mc_sig(
        self,
        wavelet,
        J,
        dj,
        dt,
        s0,
        sfunc_args1=None,
        sfunc_args2=None,
        sfunc_kwargs1=None,
        sfunc_kwargs2=None,
        mc_count=60,
        slen=None,
        sig_lvl=0.95,
        sfunc=None,
    ):
        """
        Parameters
        ----------
            wavelet : pycwt wavelet object class
            J : int
                Wavelet's maximum scale.
            dj : float
                Number of suboctaves / number of octaves for the wavelet.
            dt : float
                Spacing of the time series in time. It is recommended to use the
                spacing of the data being tested for consistenct.
            s0 : float
                Minimum resolvable scale for the wavelet. Recommended value is
                2 * dt
            sfunc_args1 : list, optional
                positional arguments for sfunc for time series one.
            sfunc_args2 : list
                positional arguments for sfunc for time series two.
            sfunc_kwargs1 : dictionary, optional
            sfunc_kwargs2 : dictionary, optional
            mc_count : int, optional
            slen : int, optional
            sig_lvl : float, optional
            sfunc : function handle
                sfunc is used to generate the synthetic data
                for the Monte-Carlo simulation. The default function is
                pycwt.rednoise(). Function must accept `N` as the first argument
                and return an array of length `N`.

        Returns
        -------
            coh : ndarray of floats
                Coherence from each Monte-Carlo draw

        """

        if sfunc_args1 is None:
            sfunc_args1 = []
        if sfunc_args2 is None:
            sfunc_args2 = []
        if sfunc_kwargs1 is None:
            sfunc_kwargs1 = {}
        if sfunc_kwargs2 is None:
            sfunc_kwargs2 = {}

        if slen is None:
            # Choose N so that largest scale has at least
            # some part outside the COI
            slen = s0 * (2 ** (J * dj)) / dt

        # Assign the length of the synthetic signal
        N = slen

        # Assign the function for generating synthetic data
        if sfunc is None:
            # @ Replace with a functional red noise generator
            sfunc = rednoise

        # Peak the details of the cwt output for a single realization
        # of the noise function.
        noise1 = sfunc(N, *sfunc_args1, **sfunc_kwargs1)

        # Check that sfunc returns an array with the necessary properties
        if not len(noise1) == N:
            raise ValueError("sfunc must return data of length N")

        nW1, sj, freq, coi, _, _ = pycwt.cwt(
            noise1, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet
        )

        period = np.ones([1, N]) / freq[:, None]
        coi = np.ones([J + 1, 1]) * coi[None, :]
        outsidecoi = period <= coi
        scales = np.ones([1, N]) * sj[:, None]
        sig_ind = np.zeros(J + 1)
        maxscale = find(outsidecoi.any(axis=1))[-1]
        sig_ind[outsidecoi.any(axis=1)] = np.nan

        coh = np.ma.zeros([J + 1, N, mc_count])

        # Displays progress bar with tqdm
        for n_mc in tqdm(range(mc_count)):  # , disable=not progress):
            # Generates a synthetic signal using the provided function and
            # parameters

            noise1 = sfunc(N, *sfunc_args1, **sfunc_kwargs1)
            noise2 = sfunc(N, *sfunc_args2, **sfunc_kwargs2)

            # Calculate the cross wavelet transform of both red-noise signals
            kwargs = dict(dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
            nW1, sj, freq, coi, _, _ = pycwt.cwt(noise1, **kwargs)
            nW2, sj, freq, coi, _, _ = pycwt.cwt(noise2, **kwargs)
            nW12 = nW1 * nW2.conj()

            # Smooth wavelet wavelet transforms and calculate wavelet coherence
            # between both signals.

            S1 = wavelet.smooth(np.abs(nW1) ** 2 / scales, dt, dj, sj)
            S2 = wavelet.smooth(np.abs(nW2) ** 2 / scales, dt, dj, sj)
            S12 = wavelet.smooth(nW12 / scales, dt, dj, sj)

            R2 = np.ma.array(np.abs(S12) ** 2 / (S1 * S2), mask=~outsidecoi)
            coh[:, :, n_mc] = R2

        period = period[:, 0]

        return coh, period, scales, coi


def coi_where(period, coi, data):
    """
    Finds where the period by num samples array is outside the coi.
    Useful for creating masked numpy arrays.

    INPUTS:
    period : N array, the wavelet periods
    coi : M array, the coi location for each sample. M should
        be the number of samples.
    data : N x M array, the data to broadcast the COI mask to.

    RETURNS:
    outside_coi : N by M array of booleans. Is True where
        the wavelet is effected by the coi.

    EXAMPLE:
        outside_coi = pycwt_stat_helpers.coi_where(period, coi, WCT)
        masked_WCT = np.ma.array(WCT, mask=~outside_coi)
    """

    coi_matrix = np.ones(np.shape(data)) * coi
    period_matrix = (np.ones(np.shape(data)).T * period).T

    outside_coi = period_matrix < coi_matrix

    return outside_coi
