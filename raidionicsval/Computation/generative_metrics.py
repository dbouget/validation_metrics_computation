import numpy as np
from scipy.fft import fft, fftfreq
from scipy.ndimage import sobel
from skimage.metrics import mean_squared_error, structural_similarity, normalized_mutual_information, peak_signal_noise_ratio


def parallel_metric_computation(args):
    """
    Metrics computation method linked to the multiprocessing strategy. Effectively where the call to compute is made.
    :param args: list of arguments split from the lists given to the multiprocessing.Pool call.
    :return: list with metric name and computed metric value.
    """
    gt = args[0]
    generative = args[1]
    metrics = args[2]
    metrics_values = args[3]


    try:
        if isinstance(metrics, list):
            for i, metric in enumerate(metrics):
                if metrics_values[i] == metrics_values[i] and metrics_values[i] is not None:
                    continue
                metric_value = compute_specific_metric_value(metric=metric, gt=gt, generative=generative)
                metrics_values[i] = metric_value
            return list(zip(metrics, metrics_values))
        else:
            if metrics_values != metrics_values or metrics_values is None:
                metric_value = compute_specific_metric_value(metric=metrics, gt=gt, generative=generative)
                metrics_values = metric_value
            return [metrics, metrics_values]
    except Exception as e:
        print(f'Computing the metrics gave the following exception: {e}')
        pass



def compute_specific_metric_value(metric, gt, generative):
    metric_value = None

    if metric == "mae":
        metric_value = np.mean(np.abs(gt - generative))
    elif metric == "mse":
        metric_value =  np.mean((gt - generative) ** 2)
    elif metric == "kl":
        """
        KL divergence between intensity histograms of GT and prediction.
        """
        hist_gt, _ = np.histogram(gt.flatten(), bins=256, range=(0, 1), density=True)
        hist_pred, _ = np.histogram(generative.flatten(), bins=256, range=(0, 1), density=True)

        epsilon = 1e-8
        hist_gt += epsilon
        hist_pred += epsilon
        kl = np.sum(hist_gt * np.log(hist_gt / hist_pred))
        metric_value = kl
    elif metric == "psnr":
        metric_value = peak_signal_noise_ratio(gt, generative, data_range=1.)
    elif metric == "ssim":
        metric_value = structural_similarity(gt, generative, data_range=1.)
    elif metric == "nmi":
        metric_value = normalized_mutual_information(gt, generative)
    elif metric == "ncc":
        g_mean = gt.mean()
        p_mean = generative.mean()
        numerator = ((gt - g_mean) * (generative - p_mean)).sum()
        denominator = np.sqrt(((gt - g_mean) ** 2).sum() * ((generative - p_mean) ** 2).sum() + 1e-8)
        metric_value = (numerator / denominator).item()
    elif metric == "flicker":
        gt_flicker = flicker_score(gt, foreground_mask(gt))
        generative_flicker = flicker_score(generative, foreground_mask(generative))
        metric_value = [gt_flicker, generative_flicker]
    elif metric == "consistency":
        gt_con = slice_consistency(gt)
        generative_con = slice_consistency(generative)
        metric_value = [gt_con, generative_con]
    return metric_value


def foreground_mask(vol, thresh=0.01):
    mask = np.max(vol, axis=2) > thresh
    return mask


def compute_ncc(gt, generative):
    g_mean = gt.mean()
    p_mean = generative.mean()
    numerator = ((gt - g_mean) * (generative - p_mean)).sum()
    denominator = np.sqrt(((gt - g_mean) ** 2).sum() * ((generative - p_mean) ** 2).sum() + 1e-8)
    metric_value = (numerator / denominator).item()
    return metric_value

def gradient_correlation(a, b):
    if len(a.shape) == len(b.shape) == 2:
        gx_a, gy_a = sobel(a, axis=0), sobel(a, axis=1)
        gx_b, gy_b = sobel(b, axis=0), sobel(b, axis=1)
        mag_a = np.sqrt(gx_a**2 + gy_a**2)
        mag_b = np.sqrt(gx_b**2 + gy_b**2)
    elif len(a.shape) == len(b.shape) == 3:
        gx_a, gy_a, gz_a = sobel(a, axis=0), sobel(a, axis=1), sobel(a, axis=2)
        gx_b, gy_b, gz_b = sobel(b, axis=0), sobel(b, axis=1), sobel(b, axis=2)
        mag_a = np.sqrt(gx_a**2 + gy_a**2 + gz_a**2)
        mag_b = np.sqrt(gx_b**2 + gy_b**2 + gz_b**2)
    return np.corrcoef(mag_a.flatten(), mag_b.flatten())[0, 1]

def slice_consistency(volume):
    """

    """
    ncc_vals, ssim_vals, grad_vals = [], [], []
    for i in range(volume.shape[2] - 1):
        a, b = volume[:, :, i], volume[:, :, i + 1]
        ncc_vals.append(compute_ncc(a, b))
        ssim_vals.append(structural_similarity(a, b, data_range=1.))
        grad_vals.append(gradient_correlation(a, b))

    return {
        "NCC": np.array(ncc_vals),
        "SSIM": np.array(ssim_vals),
        "GradientCorr": np.array(grad_vals)
    }

def slice_pair_metrics(vol, data_range=1.0):
    """
    vol: H x W x D (numpy)
    returns dict with arrays of length D-1 for L1, L2, SSIM between slice i and i+1
    """
    H,W,D = vol.shape
    l1, l2, ssim_vals = [], [], []
    for i in range(D-1):
        a = vol[:,:,i]; b = vol[:,:,i+1]
        l1.append(np.nanmean(np.abs(a-b)))
        l2.append(np.sqrt(np.nanmean((a-b)**2)))
        try:
            ssim_vals.append(structural_similarity(a, b, data_range=data_range))
        except Exception:
            ssim_vals.append(np.nan)
    return {"L1": np.array(l1), "L2": np.array(l2), "SSIM": np.array(ssim_vals)}

def z_second_derivative_energy(vol):
    """
    Compute per-slice second-derivative magnitude: |V[i+1] - 2V[i] + V[i-1]|
    returns array length D (first and last will be zero-padded)
    """
    H,W,D = vol.shape
    sec = np.zeros(D, dtype=np.float64)
    for i in range(1, D-1):
        diff2 = vol[:,:,i+1] - 2*vol[:,:,i] + vol[:,:,i-1]
        sec[i] = np.nanmean(np.abs(diff2))
    # pad edges with neighbor values (or keep zero)
    sec[0] = sec[1] if D>1 else 0.0
    sec[-1] = sec[-2] if D>1 else 0.0
    return sec


def temporal_std_map(vol, mask=None):
    """
    vol: HxWxD
    returns std_map HxW and summary stats
    """
    std_map = np.nanstd(vol, axis=2)
    if mask is not None:
        masked = std_map[mask]
    else:
        masked = std_map.flatten()
    return std_map, np.nanmean(masked), np.nanstd(masked)


def high_frequency_energy_ratio(vol, mask=None, hf_frac=0.25):
    """
    For each (x,y) compute FFT along z, compute ratio of energy in high freq band (top hf_frac)
    returns map HxW of ratios and aggregated mean/std over mask
    hf_frac: fraction of frequencies (0..0.5) considered 'high' (Nyquist at 0.5)
    """
    H,W,D = vol.shape
    # pad or remove DC? We'll compute real FFT
    freqs = fftfreq(D)  # values from 0 to fs/2 and negative
    # Use absolute frequency ordering and take positive freqs (including DC)
    pos_idx = np.where(freqs >= 0)[0]
    # threshold index for high-frequency band: take top hf_frac of positive frequencies
    n_pos = len(pos_idx)
    hf_start = int(np.floor(n_pos * (1 - hf_frac)))
    hf_idx = pos_idx[hf_start:]
    # compute FFT along z for each voxel
    ratios = np.zeros((H,W), dtype=np.float64)
    for x in range(H):
        for y in range(W):
            ts = vol[x,y,:]
            if np.all(np.isnan(ts)):
                ratios[x,y] = np.nan
                continue
            spec = np.abs(fft(ts))
            total_energy = np.sum(spec[pos_idx]**2) + 1e-12
            hf_energy = np.sum(spec[hf_idx]**2)
            ratios[x,y] = hf_energy / total_energy
    if mask is not None:
        vals = ratios[mask]
    else:
        vals = ratios.flatten()
    return ratios, np.nanmean(vals), np.nanstd(vals)


def temporal_autocorr_lag1_map(vol, mask=None):
    """
    Compute per-voxel lag-1 autocorrelation along z.
    """
    H,W,D = vol.shape
    ac_map = np.zeros((H,W), dtype=np.float64)
    for x in range(H):
        for y in range(W):
            ts = vol[x,y,:]
            if np.isnan(ts).all():
                ac_map[x,y] = np.nan
                continue
            ts = ts - np.nanmean(ts)
            denom = np.nansum(ts**2)
            if denom < 1e-12:
                ac_map[x,y] = np.nan
                continue
            num = np.nansum(ts[:-1]*ts[1:])
            ac_map[x,y] = num/denom
    if mask is not None:
        vals = ac_map[mask]
    else:
        vals = ac_map.flatten()
    return ac_map, np.nanmean(vals), np.nanstd(vals)


def flicker_score(volume, mask=None, hf_frac=0.25, weights=None):
    """
    Combine normalized measures into a single flicker score (higher -> more flicker).
    Returns dict with components and combined scalar.
    """
    vol_norm = volume.copy()
    if mask is None:
        mask2d = np.ones((volume.shape[0], volume.shape[1]), dtype=bool)
    else:
        mask2d = mask

    # 1. mean second-derivative energy (normalized)
    sec = z_second_derivative_energy(vol_norm)
    sec_masked = sec  # sec is per-slice, not per-voxel; take mean
    sec_mean = np.nanmean(sec_masked)

    # 2. mean slice-to-slice L2
    pair = slice_pair_metrics(vol_norm)
    l2_mean = np.nanmean(pair["L2"])

    # 3. high-frequency energy ratio (voxel-level)
    _, hf_mean, hf_std = high_frequency_energy_ratio(vol_norm, mask=mask2d, hf_frac=hf_frac)

    # 4. temporal std (mean)
    _, tstd_mean, tstd_std = temporal_std_map(vol_norm, mask=mask2d)

    # Normalize components to unit scale (robust by using median-absolute or simple normalization)
    # Use a simple heuristic normalization by dividing by small constants (tunable)
    c_sec = sec_mean
    c_l2 = l2_mean
    c_hf = hf_mean  # already a ratio [0,1]
    c_tstd = tstd_mean

    # weights
    if weights is None:
        weights = {"sec":1.0, "l2":1.0, "hf":1.0, "tstd":1.0}

    # combine (make sure hf in [0,1])
    combined = (weights["sec"] * c_sec) + (weights["l2"] * c_l2) + (weights["hf"] * c_hf) + (weights["tstd"] * c_tstd)
    # return all parts and the combined scalar
    return {
        "sec_mean": float(c_sec),
        "l2_mean": float(c_l2),
        "hf_mean": float(c_hf),
        "tstd_mean": float(c_tstd),
        "combined_score": float(combined),
        "pair_metrics": pair,
        "sec_per_slice": sec
    }