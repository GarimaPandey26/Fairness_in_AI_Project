import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.metrics import ClassificationMetric

# 1. Generate synthetic dataset
np.random.seed(42)
data = {
    'age_group': np.random.choice([0, 1], 1000, p=[0.5, 0.5]),
    'income': np.random.normal(50, 15, 1000)
}
data['expenditure'] = 50 + 0.5 * data['income'] + 5 * data['age_group'] + np.random.normal(0, 10, 1000)

df = pd.DataFrame(data)
df['expenditure_binary'] = (df['expenditure'] > df['expenditure'].median()).astype(int)

# 2. Train-test split
X = df[['age_group', 'income']]
y = df['expenditure_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Define privileged/unprivileged groups
privileged_groups = [{'age_group': 1}]
unprivileged_groups = [{'age_group': 0}]

# 4. Create AIF360 dataset
train_dataset = BinaryLabelDataset(df=pd.concat([X_train, y_train], axis=1),
                                   label_names=['expenditure_binary'],
                                   protected_attribute_names=['age_group'],
                                   privileged_protected_attributes=[1])

test_dataset = BinaryLabelDataset(df=pd.concat([X_test, y_test], axis=1),
                                  label_names=['expenditure_binary'],
                                  protected_attribute_names=['age_group'],
                                  privileged_protected_attributes=[1])

# 5. Baseline model
clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

test_pred_baseline = test_dataset.copy()
test_pred_baseline.labels = y_pred.reshape(-1, 1)

metric_baseline = ClassificationMetric(test_dataset, test_pred_baseline,
                                       privileged_groups=privileged_groups,
                                       unprivileged_groups=unprivileged_groups)

acc_base = accuracy_score(y_test, y_pred)
di_base = metric_baseline.disparate_impact()
spd_base = metric_baseline.statistical_parity_difference()

print("\n⚖️ Baseline Metrics:")
print(f"Accuracy: {acc_base:.2f}, DI: {di_base:.2f}, SPD: {spd_base:.2f}")

# 6. Reweighing
rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
train_dataset_rw = rw.fit_transform(train_dataset)

clf_rw = LogisticRegression(solver='liblinear')
clf_rw.fit(X_train, y_train, sample_weight=train_dataset_rw.instance_weights)
y_pred_rw = clf_rw.predict(X_test)

test_pred_rw = test_dataset.copy()
test_pred_rw.labels = y_pred_rw.reshape(-1, 1)

metric_rw = ClassificationMetric(test_dataset, test_pred_rw,
                                 privileged_groups=privileged_groups,
                                 unprivileged_groups=unprivileged_groups)

acc_rw = accuracy_score(y_test, y_pred_rw)
di_rw = metric_rw.disparate_impact()
spd_rw = metric_rw.statistical_parity_difference()

print("\n⚖️ After Reweighing:")
print(f"Accuracy: {acc_rw:.2f}, DI: {di_rw:.2f}, SPD: {spd_rw:.2f}")

# 7. ROC Post-processing
roc = RejectOptionClassification(privileged_groups=privileged_groups,
                                 unprivileged_groups=unprivileged_groups,
                                 low_class_thresh=0.01,
                                 high_class_thresh=0.99,
                                 num_class_thresh=100,
                                 num_ROC_margin=50,
                                 metric_name="Statistical parity difference",
                                 metric_ub=0.05,
                                 metric_lb=-0.05)

roc.fit(test_dataset, test_pred_baseline)
test_pred_roc = roc.predict(test_pred_baseline)

metric_roc = ClassificationMetric(test_dataset, test_pred_roc,
                                  privileged_groups=privileged_groups,
                                  unprivileged_groups=unprivileged_groups)

acc_roc = accuracy_score(y_test, test_pred_roc.labels.ravel())
di_roc = metric_roc.disparate_impact()
spd_roc = metric_roc.statistical_parity_difference()

print("\n⚖️ After ROC:")
print(f"Accuracy: {acc_roc:.2f}, DI: {di_roc:.2f}, SPD: {spd_roc:.2f}")

# 8. Visualization
labels = ['SPD', 'DI', 'Accuracy']
baseline = [spd_base, di_base, acc_base]
reweighing = [spd_rw, di_rw, acc_rw]
roc_values = [spd_roc, di_roc, acc_roc]

x = np.arange(len(labels))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - width, baseline, width, label='Baseline', color='royalblue')
plt.bar(x, reweighing, width, label='Reweighing', color='darkorange')
plt.bar(x + width, roc_values, width, label='ROC', color='forestgreen')

plt.axhline(y=1.0, linestyle='--', color='gray', linewidth=1, label='Ideal DI')
plt.axhline(y=0.0, linestyle='--', color='black', linewidth=1, label='Ideal SPD')

plt.xticks(x, labels)
plt.ylabel('Metric Value')
plt.title('Fairness Metrics on Medical Expenditure Dataset')
plt.legend()
plt.tight_layout()
plt.show()
