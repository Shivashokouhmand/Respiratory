"""# please note that for the first part of the project we extracted many fetures ,and In our paper we used feature
# importance for our feature selection part. then after finding the features with the highest F-score in this code
# we removed extra features.
# Also, we used several hyperparameters and employed Gridsearch to find the best params, here in order to decrease the
# run time we used what we find is the best and omit the Gridsearch part."""




from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from typing import Iterable
import scipy.signal as signal
import librosa
from scipy.stats import kurtosis, skew
from scipy.signal import get_window
import pandas as pd
import matplotlib.pyplot as plt
import shap




def butter_bandpass_filter(data,sr):

    lowcut = 50
    highcut = 2500
    nyquist_rate = sr / 2.0
    lowcut_norm = lowcut / nyquist_rate
    highcut_norm = highcut / nyquist_rate
    order = 5

    b, a = signal.butter(order, [lowcut_norm, highcut_norm], btype='band', analog=False)
    y_filtered = signal.lfilter(b, a, data)
    return y_filtered




# FFT
def signal_fft(signal):
    sig_fft = np.fft.fft(signal)
    energy_total = sum(abs(sig_fft) ** 2)
    sig_fft = sig_fft[:int(len(signal) / 2)]

    signal_chunks = np.array_split(sig_fft, 20)  #
    ratio = []
    for chunk in signal_chunks:
        energy_chunk = sum(abs(chunk) ** 2)
        ratio.append(energy_chunk / energy_total)

    return np.array(ratio)


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def extract_feature(signal, magnitude=True, centroid=True, zero_crossing_rate=True, mfcc=True, rolloff=False, diff=False):
    # stereo to mono
    if len(signal.shape) > 1:
        signal = signal[:, 0]

    else:
        pass

    s_rate = 6000

    if magnitude:
        n_fft = int(s_rate * 0.02)
        hop_length = n_fft // 2
        pitch, magnitude = librosa.piptrack(y=signal, sr=s_rate, n_fft=n_fft, hop_length=hop_length)
        magnitude_mean = np.mean(magnitude)
        magnitude_max = np.max(magnitude)
        magnitude_std = np.std(magnitude)
        magnitude_skew = skew(list(flatten(magnitude)))
        magnitude_kurtosis = kurtosis(list(flatten(magnitude)))

        magnitude = np.array((magnitude_mean, magnitude_max,  magnitude_std), dtype=object)

    if diff:
        signal_diff = np.diff(signal)
        diff_mean = np.mean(signal_diff)
        diff_max = np.max(signal_diff)
        diff_min = np.min(signal_diff)
        diff_median = np.median(signal_diff)
        diff_std = np.std(signal_diff)
        diff_skew = skew(list(flatten(signal_diff)))
        diff_kurtosis = kurtosis(list(flatten(signal_diff)))
        diff_diff = np.diff(signal_diff)
        diff2_mean = np.mean(diff_diff)
        diff2_max = np.max(diff_diff)
        diff2_min = np.min(diff_diff)
        diff2_median = np.median(diff_diff)
        diff2_std = np.std(diff_diff)
        diff2_skew = skew(list(flatten(diff_diff)))
        diff2_kurtosis = kurtosis(list(flatten(diff_diff)))
        diff_all = np.array((diff_mean, diff_max, diff_min, diff_median, diff_std, diff_skew, diff_kurtosis, diff2_mean,
                            diff2_max, diff2_min, diff2_median, diff2_std, diff2_skew, diff2_kurtosis))

    if rolloff:
        rolloff = librosa.feature.spectral_rolloff(S=s, sr=s_rate,**args )
        rolloff_mean = np.mean(rolloff)
        rolloff_max = np.max(rolloff)
        rolloff_min = np.min(rolloff)
        rolloff_median = np.median(rolloff)
        rolloff_std = np.std(rolloff)
        rolloff_skew = skew(list(flatten(rolloff)))
        rolloff_kurtosis = kurtosis(list(flatten(rolloff)))
        roll_off = np.array((rolloff_mean, rolloff_max, rolloff_min, rolloff_median, rolloff_std, rolloff_skew,
                             rolloff_kurtosis), dtype=object)


    if zero_crossing_rate:
        zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=n_fft, hop_length=hop_length)
        zcr_mean = np.mean(zcr)
        zcr_max = np.max(zcr)
        zcr_min = np.min(zcr)
        zcr_median = np.median(zcr)
        zcr_std = np.std(zcr)
        zcr_skew = skew(list(flatten(zcr)))
        zcr_kurtosis = kurtosis(list(flatten(zcr)))
        zero_rate = np.array((zcr_mean, zcr_max, zcr_min, zcr_median, zcr_std), dtype=object)


    s, phase = librosa.magphase(librosa.stft(signal))
    if centroid:
        centroid = librosa.feature.spectral_centroid(S=s)
        centroid_mean = np.mean(centroid)
        centroid_max = np.max(centroid)
        centroid_min = np.min(centroid)
        centroid_median = np.median(centroid)
        centroid_std = np.std(centroid)
        centroid_skew = skew(list(flatten(centroid)))
        centroid_kurtosis = kurtosis(list(flatten(centroid)))
        center = np.array((centroid_mean, centroid_max, centroid_min, centroid_median, centroid_std), dtype=object)

    if mfcc:
        meltspec_args = {"n_fft": n_fft, "hop_length": hop_length, "window": get_window("hamming", 120)}
        mfcc = librosa.feature.mfcc(y=signal, sr=s_rate, S=None, n_mfcc=13, **meltspec_args).T
        feature_mean = []
        feature_max = []
        feature_min = []
        feature_std = []
        feature_median = []
        for i in range(0, 13):
            mfcc_mean = np.mean(mfcc[:, i])
            mfcc_max = np.max(mfcc[:, i])
            mfcc_min = np.min(mfcc[:, i])
            mfcc_median = np.median(mfcc[:, i])
            mfcc_std = np.std(mfcc[:, i])
            feature_mean.append(mfcc_mean)
            feature_max.append(mfcc_max)
            feature_min.append(mfcc_min)
            feature_median.append(mfcc_median)
            feature_std.append(mfcc_std)

        mfcc_skew = skew(list(flatten(mfcc)))
        mfcc_kurtosis = kurtosis(list(flatten(mfcc)))
        delta_mfcc = librosa.feature.delta(list(flatten(mfcc)))
        delta_mfcc_mean = np.mean(delta_mfcc)
        delta_mfcc_max = np.max(delta_mfcc)
        delta_mfcc_min = np.min(delta_mfcc)
        delta_mfcc2 = librosa.feature.delta(list(flatten(mfcc)), order=2)
        delta_mfcc2_mean = np.mean(delta_mfcc2)
        delta_mfcc2_max = np.max(delta_mfcc2)
        delta_mfcc2_min = np.min(delta_mfcc2)

        mfcdelta = np.array((feature_mean[0], feature_mean[1], feature_mean[2], feature_mean[3], feature_mean[4],
                             feature_mean[5], feature_mean[6],  feature_mean[7], feature_mean[8],
                             feature_mean[9], feature_mean[10], feature_mean[11], feature_mean[12],

                             feature_max[0],  feature_max[1], feature_max[2], feature_max[3], feature_max[4],
                             feature_max[5], feature_max[6],  feature_max[7], feature_max[8],
                             feature_max[9], feature_max[10], feature_max[11], feature_max[12],

                             feature_min[0], feature_min[1], feature_min[2], feature_min[3], feature_min[4],
                             feature_min[5], feature_min[6],  feature_min[7], feature_min[8],
                             feature_min[9], feature_min[10], feature_min[11], feature_min[12],

                             feature_median[0], feature_median[1], feature_median[2], feature_median[3], feature_median[4],
                             feature_median[5], feature_median[6],  feature_median[7], feature_median[8],
                             feature_median[9], feature_median[10], feature_median[11], feature_median[12],

                             feature_std[0], feature_std[1], feature_std[2], feature_std[3], feature_std[4],
                             feature_std[5], feature_std[6], feature_std[7], feature_std[8],
                             feature_std[9], feature_std[10], feature_std[11], feature_std[12]), dtype=object)

        return magnitude, zero_rate, center, mfcdelta

