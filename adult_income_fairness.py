import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from aif360.datasets import AdultDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.metrics import ClassificationMetric
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.datasets import BinaryLabelDataset

# Load dataset
dataset = AdultDataset()

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

# Split data
train, test = dataset.split([0.7], shuffle=True)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(train.features)
X_test = scaler.transform(test.features)

# -------------------------
# 1️⃣ Baseline Model
# -------------------------
clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, train.labels.ravel())
y_pred = clf.predict(X_test)

test_pred = test.copy()
test_pred.labels = y_pred.reshape(-1, 1)

metric_orig = ClassificationMetric(test, test_pred, unprivileged_groups, privileged_groups)

acc_orig = accuracy_score(test.labels, y_pred)
spd_orig = metric_orig.statistical_parity_difference()
di_orig = metric_orig.disparate_impact()

print("⚖️ Baseline Metrics:")
print(f"Accuracy: {acc_orig:.2f}, DI: {di_orig:.2f}, SPD: {spd_orig:.2f}")

# -------------------------
# 2️⃣ Reweighing
# -------------------------
RW = Reweighing(unprivileged_groups, privileged_groups)
train_reweighed = RW.fit_transform(train)

clf_rw = LogisticRegression(solver='liblinear')
clf_rw.fit(scaler.transform(train_reweighed.features), train_reweighed.labels.ravel(), sample_weight=train_reweighed.instance_weights)

y_pred_rw = clf_rw.predict(X_test)
test_pred_rw = test.copy()
test_pred_rw.labels = y_pred_rw.reshape(-1, 1)

metric_rw = ClassificationMetric(test, test_pred_rw, unprivileged_groups, privileged_groups)

acc_rw = accuracy_score(test.labels, y_pred_rw)
spd_rw = metric_rw.statistical_parity_difference()
di_rw = metric_rw.disparate_impact()

print("\n⚖️ After Reweighing:")
print(f"Accuracy: {acc_rw:.2f}, DI: {di_rw:.2f}, SPD: {spd_rw:.2f}")

# -------------------------
# 3️⃣ Reject Option Classification (ROC)
# -------------------------
ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups,
                                 low_class_thresh=0.01, high_class_thresh=0.99,
                                 num_class_thresh=100, num_ROC_margin=50,
                                 metric_name="Statistical parity difference",
                                 metric_ub=0.05, metric_lb=-0.05)

ROC = ROC.fit(test, test_pred)
roc_pred = ROC.predict(test_pred)

metric_roc = ClassificationMetric(test, roc_pred, unprivileged_groups, privileged_groups)
acc_roc = accuracy_score(test.labels, roc_pred.labels)
spd_roc = metric_roc.statistical_parity_difference()
di_roc = metric_roc.disparate_impact()

print("\n⚖️ After ROC:")
print(f"Accuracy: {acc_roc:.2f}, DI: {di_roc:.2f}, SPD: {spd_roc:.2f}")

# -------------------------
# 4️⃣ Visualization
# -------------------------
metrics = ['SPD', 'DI', 'Accuracy']
baseline = [spd_orig, di_orig, acc_orig]
reweighing = [spd_rw, di_rw, acc_rw]
roc = [spd_roc, di_roc, acc_roc]

bar_width = 0.2
x = np.arange(len(metrics))

plt.figure(figsize=(10, 6))
plt.bar(x, baseline, width=bar_width, label='Baseline', color='royalblue')
plt.bar(x + bar_width, reweighing, width=bar_width, label='Reweighing', color='darkorange')
plt.bar(x + 2 * bar_width, roc, width=bar_width, label='ROC', color='seagreen')

plt.axhline(y=0.0, color='black', linestyle='--')
plt.axhline(y=1.0, color='gray', linestyle='--', label='Ideal DI')

plt.xticks(x + bar_width, metrics)
plt.ylabel('Metric Value')
plt.title('Fairness Metrics on Adult Income Dataset')
plt.legend()
plt.tight_layout()
plt.show()
