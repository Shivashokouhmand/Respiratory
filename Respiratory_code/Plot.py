from module import *
from Classifier import *



# Explain model predictions using SHAP values
# Since the name of the features would be considered we should drop them from X
data = pd.read_csv('feature_vector.csv')
X1 = data.drop(['label', 'Unnamed: 0'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X1)
model.fit(X_scaled, y)

explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X_scaled)


# Summary plot
shap.summary_plot(shap_values, X_scaled, feature_names=X1.columns, show=False)
plt.title('Summary Plot of SHAP Values')
plt.show()

# Dependence Plot
feature_to_plot = "mfcc_std_3"
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
axes = axes.flatten()

for class_idx, class_name in enumerate(y.unique()):
    shap.dependence_plot(feature_to_plot, shap_values[class_idx], X_scaled, feature_names=X1.columns, display_features=X1, show=False, ax=axes[class_idx])
    axes[class_idx].set_title(f'Dependence Plot (Class: {class_name})', fontsize=12)
    axes[class_idx].set_xlabel(f'{feature_to_plot} Feature Value', fontsize=10)
    axes[class_idx].set_ylabel('SHAP Value', fontsize=10)

# Remove empty subplot if the number of classes is less than 4
#if len(y.unique()) < 4:
    #fig.delaxes(axes[-1])
#plt.tight_layout()
plt.show()