from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import matplotlib.pyplot as plt


def confusion_matrix_plot(gt, pred, classes, output_dir):
    cm = confusion_matrix(gt, pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax = sns.heatmap(cm, cmap='YlGn', xticklabels=classes, yticklabels=classes,
                     cbar=False, annot=True, fmt='.0f')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_yticklabels(classes, rotation=0)
    output_filename = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(output_filename)
