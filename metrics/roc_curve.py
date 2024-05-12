import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def create_roc_curve(y_test, y_pred, model_name, use_case_name):
  fpr, tpr, _ = roc_curve(y_test, y_pred)
  plt.figure()
  plt.plot(fpr, tpr, label=f'ROC curve (area = {auc(fpr, tpr):.2f})')
  plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f'ROC Curve for {model_name} in {use_case_name}')
  plt.legend(loc="lower right")

  # Save and return plot to a variable
  roc_plot = plt.gcf()
  return roc_plot

