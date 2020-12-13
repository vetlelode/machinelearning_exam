from sklearn.metrics import confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt


def plotConfusion(real_values, pred_values) -> None:
    skplt.metrics.plot_confusion_matrix(
        real_values, pred_values, figsize=(8, 8))
    plt.show()
