import numpy as np
import matplotlib.pyplot as plt
from aif360.datasets import GermanDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import RejectOptionClassification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load dataset
data = GermanDataset()
X = data.features
y = data.labels.ravel()

privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

# Baseline
clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=1000))
clf.fit(X, y)
y_pred = clf.predict(X)

data_pred = data.copy()
data_pred.labels = y_pred.reshape(-1, 1)

metric = ClassificationMetric(data, data_pred,
                               privileged_groups=privileged_groups,
                               unprivileged_groups=unprivileged_groups)
spd = metric.statistical_parity_difference()
di = metric.disparate_impact()
eod = metric.equal_opportunity_difference()
aod = metric.average_odds_difference()
print(f"\n⚖️ Baseline Metrics:\nSPD: {spd:.4f}, DI: {di:.4f}, EOD: {eod:.4f}, AOD: {aod:.4f}")

# Reweighing
rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
rw.fit(data)
data_transf = rw.transform(data)

clf_rw = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=1000))
clf_rw.fit(X, y, logisticregression__sample_weight=data_transf.instance_weights)
y_pred_rw = clf_rw.predict(X)

data_pred_rw = data.copy()
data_pred_rw.labels = y_pred_rw.reshape(-1, 1)

metric_rw = ClassificationMetric(data, data_pred_rw,
                                  privileged_groups=privileged_groups,
                                  unprivileged_groups=unprivileged_groups)
print(f"\n⚖️ After Reweighing:\nSPD: {metric_rw.statistical_parity_difference():.4f}, "
      f"DI: {metric_rw.disparate_impact():.4f}, "
      f"EOD: {metric_rw.equal_opportunity_difference():.4f}, "
      f"AOD: {metric_rw.average_odds_difference():.4f}")

# ROC Post-processing
roc = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
roc.fit(data, data_pred)
data_roc = roc.predict(data_pred)

metric_roc = ClassificationMetric(data, data_roc,
                                  privileged_groups=privileged_groups,
                                  unprivileged_groups=unprivileged_groups)
print(f"\n⚖️ After ROC:\nSPD: {metric_roc.statistical_parity_difference():.4f}, "
      f"DI: {metric_roc.disparate_impact():.4f}, "
      f"EOD: {metric_roc.equal_opportunity_difference():.4f}, "
      f"AOD: {metric_roc.average_odds_difference():.4f}")

# Plotting
labels = ['SPD', 'DI', 'EOD', 'AOD']
baseline_metrics = [spd, di, eod, aod]
rw_metrics = [metric_rw.statistical_parity_difference(), metric_rw.disparate_impact(),
              metric_rw.equal_opportunity_difference(), metric_rw.average_odds_difference()]
roc_metrics = [metric_roc.statistical_parity_difference(), metric_roc.disparate_impact(),
               metric_roc.equal_opportunity_difference(), metric_roc.average_odds_difference()]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, baseline_metrics, width, label='Baseline')
plt.bar(x, rw_metrics, width, label='Reweighing')
plt.bar(x + width, roc_metrics, width, label='ROC')

plt.ylabel('Metric Value')
plt.title('Fairness Metrics on German Credit Dataset')
plt.xticks(x, labels)
plt.legend()
plt.tight_layout()
plt.show()
