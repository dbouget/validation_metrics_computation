import pandas as pd
import os
import matplotlib.pyplot as plt


def plot_fold_average(folder):
    fold_average_df = pd.read_csv(os.path.join(folder, "folds_metrics_average.csv"))

