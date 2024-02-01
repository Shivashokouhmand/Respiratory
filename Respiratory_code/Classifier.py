from module import *

# Set a random seed it could be any numbers (Thi would affect the result)
random_seed = 25

data = pd.read_csv('feature_vector.csv')

X = data.drop(['label'], axis=1)
y = data['label']


all_preds = np.array([])
all_actual = np.array([])
model = XGBClassifier(random_state=random_seed)
# Set up the Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
# Perform cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Data augmentation by adding random noise
    noise = np.random.randn(X_train_scaled.shape[0], X_train_scaled.shape[1])
    augmented_data = X_train_scaled + 0.2 * noise
    X_train_augmented = np.concatenate((X_train_scaled, augmented_data), axis=0)
    y_train_augmented = np.concatenate((y_train, y_train), axis=0)
    model.fit(X_train_augmented, y_train_augmented)
    preds = model.predict(X_test_scaled)
    all_preds = np.concatenate((all_preds, preds))
    all_actual = np.concatenate((all_actual, np.array(y_test)))

# Save the numpy files
np.save('predict.npy', all_preds)
np.save('actual.npy', all_actual)
preds = np.load('predict.npy')
actual = np.load('actual.npy')

# Performance evaluation
cm = confusion_matrix(actual, preds)
tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

accuracy = accuracy_score(actual, preds)
f1 = f1_score(actual, preds, average='macro')
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ICBHI_score = (sensitivity + specificity) / 2
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("ICBHI score:", ICBHI_score)

