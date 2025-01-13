from module import *

# Use the path where you store the dataset
path_to_dataset = ...

# label map
label_map = {'(0, 0)': 0, '(0, 1)': 1, '(1, 0)': 2, '(1, 1)': 3}

# Initialize lists to store features and labels
features = []
labels = []
max_feature_length = 1000  # Adjust the maximum feature length as needed

# Read audio files and extract features
for subdir, dirs, files in os.walk(path_to_dataset):
    for each_file in files:
        if each_file.endswith(".wav"):
            audio_path = os.path.join(subdir, each_file)
            file, sr = librosa.load(audio_path)
            file = butter_bandpass_filter(file, sr=6000)

            mel_spectrogram, mfccs = compute_features(file, sr)

            # Pad or truncate the features to a fixed length
            padded_mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max(0, max_feature_length - mel_spectrogram.shape[1]))), mode='constant')

            # Flatten the features and concatenate them
            combined_features = np.concatenate([padded_mel_spectrogram.flatten(), mfccs.flatten()])

            features.append(combined_features)

            label_path = audio_path.replace(".wav", ".txt")
            # Assuming the label file has only one line corresponding to the entire audio file
            with open(label_path, "r") as f:
                line = f.readline()
                start_time, end_time, crackles, wheezes = line.split()
                crackles = int(crackles)
                wheezes = int(wheezes)
                label = (crackles, wheezes)
                label = label_map[str(label)]  # Convert label to the desired format
                labels.append(label)

Feature_space_unpacked = []
Feature_space = []
for i, sample in enumerate(recordings):
    print(i)
    feat_tmp = []
    feature = extract_feature(sample)
    Feature_space.append(feature)
    for feat in feature:
        feat_tmp += feat.tolist()
    Feature_space_unpacked.append(feat_tmp)


# Adjust column names and number of them based on the features you activate to
columns = ['magnitudes_mean', 'magnitudes_max', 'magnitudes_std',
               'magnitudes_skew', 'magnitudes_kurtosis',

               'zcr_mean', 'zcr_max', 'zcr_min', 'zcr_median', 'zcr_std', 'zcr_skew', 'zcr_kurtosis',

               'sc_centroid_mean', 'sc_centroid_max', 'sc_centroid_min', 'sc_centroid_median', 'sc_centroid_std',
               'sc_centroid_skew', 'sc_centroid_kurtosis',

               'mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3', 'mfcc_mean_4', 'mfcc_mean_5', 'mfcc_mean_6', 'mfcc_mean_7',
               'mfcc_mean_8', 'mfcc_mean_9', 'mfcc_mean_10', 'mfcc_mean_11', 'mfcc_mean_12', 'mfcc_mean_13',
               'mfcc_mean_14', 'mfcc_mean_15', 'mfcc_mean_16', 'mfcc_mean_17', 'mfcc_mean_18', 'mfcc_mean_19',
               'mfcc_mean_20', 'mfcc_mean_21', 'mfcc_mean_22', 'mfcc_mean_23', 'mfcc_mean_24', 'mfcc_mean_25',
               'mfcc_mean_26',

               'mfcc_max_1', 'mfcc_max_2', 'mfcc_max_3', 'mfcc_max_4', 'mfcc_max_5', 'mfcc_max_6', 'mfcc_max_7',
               'mfcc_max_8', 'mfcc_max_9', 'mfcc_max_10', 'mfcc_max_11', 'mfcc_max_12', 'mfcc_max_13', 'mfcc_max_14',
               'mfcc_max_15', 'mfcc_max_16', 'mfcc_max_17', 'mfcc_max_18', 'mfcc_max_19', 'mfcc_max_20', 'mfcc_max_21',
               'mfcc_max_22', ' mfcc_max_23', 'mfcc_max_24', 'mfcc_max_25', 'mfcc_max_26',

               'mfcc_min_1', 'mfcc_min_2', 'mfcc_min_3', 'mfcc_min_4', 'mfcc_min_5', 'mfcc_min_6', 'mfcc_min_7',
               'mfcc_min_8', 'mfcc_min_9', 'mfcc_min_10', 'mfcc_min_11', 'mfcc_min_12', 'mfcc_min_13', 'mfcc_min_14',
               'mfcc_min_15', 'mfcc_min_16', 'mfcc_min_17', 'mfcc_min_18', 'mfcc_min_19', 'mfcc_min_20', 'mfcc_min_21',
               'mfcc_min_22', 'mfcc_min_23', 'mfcc_min_24', 'mfcc_min_25', 'mfcc_min_26',
               'mfcc_median_1', 'mfcc_median_2', 'mfcc_median_3', 'mfcc_median_4', 'mfcc_median_5', ' mfcc_median_6',
               'mfcc_median_7', 'mfcc_median_8', 'mfcc_median_9', 'mfcc_median_10', 'mfcc_median_11', 'mfcc_median_12',
               'mfcc_median_13', 'mfcc_median_14', 'mfcc_median_15', 'mfcc_median_16', 'mfcc_median_17',
               'mfcc_median_18', ' mfcc_median_19', 'mfcc_median_20', 'mfcc_median_21', 'mfcc_median_22',
               'mfcc_median_23', 'mfcc_median_24', 'mfcc_median_25', 'mfcc_median_26',
               'mfcc_std_1', 'mfcc_std_2', 'mfcc_std_3', 'mfcc_std_4', 'mfcc_std_5', 'mfcc_std_6', 'mfcc_std_7',
               'mfcc_std_8', 'mfcc_std_9', 'mfcc_std_10', 'mfcc_std_11', 'mfcc_std_12', 'mfcc_std_13', 'mfcc_std_14',
               'mfcc_std_15', 'mfcc_std_16', 'mfcc_std_17', 'mfcc_std_18', 'mfcc_std_19', 'mfcc_std_20', 'mfcc_std_21',
               ' mfcc_std_22', ' mfcc_std_23', ' mfcc_std_24', ' mfcc_std_25', 'mfcc_std_26',
               'mfcc_skew', 'mfcc_kurtosis',
               'delta_mfcc_mean', 'delta_mfcc_max', 'delta_mfcc_min', 'delta_mfcc2_mean', 'delta_mfcc2_max',
               'delta_mfcc2_min',
               ]
df = pd.DataFrame(index=range(len(recordings)), columns=columns)

for i in range(len(Feature_space_unpacked)):  # Subject
  df.iloc[i, :] = Feature_space_unpacked[i]

df['label'] = labels

df.to_csv('feature_vector.csv')