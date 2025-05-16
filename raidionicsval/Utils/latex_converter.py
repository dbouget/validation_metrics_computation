import os
import numpy as np
import pandas as pd
from typing import List


def export_df_to_latex(folder, data, suffix=''):
    matrix_filename = os.path.join(folder, 'df_latex.txt') if suffix == '' else os.path.join(folder, 'df_' + suffix + '_latex.txt')
    columns = data.columns.values
    pfile = open(matrix_filename, 'w')
    pfile.write('\\begin{table}[h]\n')
    pfile.write('\\adjustbox{max width=\\textwidth}{\n')
    pfile.write('\\begin{tabular}{l'+('r' * int(len(columns) / 2)) + '}\n')
    pfile.write('\\toprule\n')
    header_line = '{}'
    for elem in columns:
        header_line = header_line + ' & ' + elem
    pfile.write(header_line + '\\tabularnewline\n')
    for index, row in data.iterrows():
        line = str(int(row[columns[0]])) + ' & ' + str(int(row[columns[1]]))
        for c in range(2, len(columns), 1):
            value = row[columns[c]]
            line = line + ' & $' + str(np.round(value * 100., 2)) + '$'
        pfile.write(line + '\\tabularnewline\n')
    pfile.write('\\bottomrule\n')
    pfile.write('\\end{tabular}\n')
    pfile.write('}\n')
    pfile.write('\\end{table}')
    pfile.close()


def export_segmentation_df_to_latex_paper(folder: str, class_name: str, study: str = "", categories: List = [],
                                          input_csv_filename: str = None, suffix: str = "") -> None:
    """
    For exporting the segmentation performances into a compatible latex table.

    Parameters
    ----------
    folder: str
        Main destination folder where the latex table will be saved as a text file.
    class_name: str
        Name of the structure of interest in order to load the proper performance results files.
    study: str
        Name of the specific study to generate a latex table for, used to create the destination file.
    categories: List
        If the latex is linked to a specific study, the categories contain the names of the different elements to read
        the results files from disk.
    suffix: str

    """
    columns = ['Fold', '\# Samples', 'Recall', 'Precision', 'Specificity', 'bAcc', 'Dice', 'Recall', 'Precision',
               'Dice', 'Recall', 'Precision']
    df_columns = ['Patient-wise recall (Mean)', 'Patient-wise recall (Std)', 'Patient-wise precision (Mean)',
                  'Patient-wise precision (Std)',
                  'Patient-wise specificity (Mean)', 'Patient-wise specificity (Std)',
                  'Patient-wise Balanced accuracy (Mean)', 'Patient-wise Balanced accuracy (Std)',
                  'PiW Dice (Mean)', 'PiW Dice (Std)', 'PiW Recall (Mean)', 'PiW Recall (Std)', 'PiW Precision (Mean)',
                  'PiW Precision (Std)',
                  'OW Dice (Mean)', 'OW Dice (Std)', 'OW Recall (Mean)', 'OW Recall (Std)', 'OW Precision (Mean)',
                  'OW Precision (Std)']

    if study == "":
        overall_metrics_filename = os.path.join(folder, "Validation", class_name + '_overall_metrics_average.csv') if suffix == "All" else os.path.join(
            folder, 'Validation', class_name + '_overall_metrics_average_' + suffix + '.csv')
        overall_metrics_df = pd.read_csv(overall_metrics_filename)
        matrix_filename = os.path.join(folder, 'Validation',
                                       class_name + '_overall_metrics_latex.txt') if suffix == "" else os.path.join(
            folder, 'Validation', class_name + '_overall_metrics_latex' + suffix + '.txt')
        pfile = open(matrix_filename, 'w')
        pfile.write('\\begin{table}[h]\n')
        pfile.write('\\adjustbox{max width=\\textwidth}{\n')
        pfile.write('\\begin{tabular}{rr||cccc||ccc||ccc}\n')
        pfile.write(' & & \multicolumn{4}{c||}{Patient-wise} & \multicolumn{3}{c||}{Pixel-wise} & \multicolumn{3}{c}{Object-wise} \\tabularnewline\n')
        header_line = ''
        for elem in columns[:-1]:
            header_line = header_line + elem + ' & '
        header_line = header_line + columns[-1]
        pfile.write(header_line + '\\tabularnewline\n')
        line = str(int(overall_metrics_df['Fold'].values[0])) + ' & ' + str(
            int(overall_metrics_df['# samples'].values[0]))
        for c in range(0, len(df_columns[:8]), 2):
            line = line + ' & ${:05.2f}\pm{:05.2f}$'.format(
                np.round(overall_metrics_df[df_columns[c]].values[0] * 100., 2),
                np.round(overall_metrics_df[df_columns[c + 1]].values[0] * 100., 2))
        for c in range(8, len(df_columns), 2):
            line = line + ' & ${:05.2f}\pm{:05.2f}$'.format(np.round(overall_metrics_df[df_columns[c]].values[0] * 100., 2),
                                                            np.round(overall_metrics_df[df_columns[c + 1]].values[0] * 100.,
                                                                     2))
        pfile.write(line + '\\tabularnewline\n')
        pfile.write('\\end{tabular}\n')
        pfile.write('}\n')
        pfile.write('\\end{table}')
        pfile.close()

    elif input_csv_filename is not None:
        matrix_filename = os.path.join(os.path.dirname(input_csv_filename), os.path.basename(input_csv_filename).replace('.csv', '_latex_table.txt'))
        pfile = open(matrix_filename, 'w')
        pfile.write('\\begin{table}[h]\n')
        pfile.write('\\adjustbox{max width=\\textwidth}{\n')
        pfile.write('\\begin{tabular}{rr||cccc||ccc||ccc}\n')
        pfile.write(
            ' & & \multicolumn{4}{c||}{Patient-wise} & \multicolumn{3}{c||}{Pixel-wise} & \multicolumn{3}{c}{Object-wise} \\tabularnewline\n')
        header_line = ''
        for elem in columns[:-1]:
            header_line = header_line + elem + ' & '
        header_line = header_line + columns[-1]
        pfile.write(header_line + '\\tabularnewline\n')
        overall_metrics_df = pd.read_csv(input_csv_filename)
        line = ' & ' + str(int(overall_metrics_df['# samples'].values[0]))
        for c in range(0, len(df_columns), 2):
            line = line + ' & ${:05.2f}\pm{:05.2f}$'.format(
                np.round(overall_metrics_df[df_columns[c]].values[0] * 100., 2),
                np.round(overall_metrics_df[df_columns[c + 1]].values[0] * 100., 2))
        pfile.write(line + '\\tabularnewline\n')
        pfile.write('\\end{tabular}\n')
        pfile.write('}\n')
        pfile.write('\\end{table}')
        pfile.close()
    else:
        study_path = os.path.join(folder, study)
        matrix_filename = os.path.join(study_path, class_name + '_overall_metrics_average_GT_volume_(ml)_latex.txt') if suffix == "" else os.path.join(study_path, class_name + '_overall_metrics_average_' + suffix + 'latex.txt')
        pfile = open(matrix_filename, 'w')
        pfile.write('\\begin{table}[h]\n')
        pfile.write('\\adjustbox{max width=\\textwidth}{\n')
        pfile.write('\\begin{tabular}{rr||cccc||ccc||ccc}\n')
        pfile.write(
            ' & & \multicolumn{4}{c||}{Patient-wise} & \multicolumn{3}{c||}{Pixel-wise} & \multicolumn{3}{c}{Object-wise} \\tabularnewline\n')
        header_line = ''
        for elem in columns[:-1]:
            header_line = header_line + elem + ' & '
        header_line = header_line + columns[-1]
        pfile.write(header_line + '\\tabularnewline\n')
        for cat in categories:
            overall_metrics_filename = os.path.join(study_path, class_name + '_overall_metrics_average__GT volume (ml)_' + cat + '.csv')  if suffix == "" else os.path.join(study_path, class_name + '_overall_metrics_average_' + suffix + cat + '.csv')
            overall_metrics_df = pd.read_csv(overall_metrics_filename)
            line = cat + ' & ' + str(int(overall_metrics_df['# samples'].values[0]))
            for c in range(0, len(df_columns), 2):
                line = line + ' & ${:05.2f}\pm{:05.2f}$'.format(
                    np.round(overall_metrics_df[df_columns[c]].values[0] * 100., 2),
                    np.round(overall_metrics_df[df_columns[c + 1]].values[0] * 100., 2))
            pfile.write(line + '\\tabularnewline\n')
        pfile.write('\\end{tabular}\n')
        pfile.write('}\n')
        pfile.write('\\end{table}')
        pfile.close()

def export_df_to_latex_paper_old(folder, data, suffix=''):
    """
    TO DEPRECATE!
    """
    matrix_filename = os.path.join(folder, 'df_latex_paper.txt') if suffix == '' else \
        os.path.join(folder, 'df_' + suffix + '_latex_paper.txt')
    columns = ['Fold', '\# Samples', 'Dice', 'Dice-TP', 'F1-score', 'Recall', 'Precision', 'F1-score', 'Recall', 'Precision']
    df_columns = ['Patient-wise F1', 'Patient-wise recall', 'Patient-wise precision', 'Object-wise F1', 'Object-wise recall', 'Object-wise precision'] #'Fold', '# Samples', 'Dice', 'Dice-TP',
    pfile = open(matrix_filename, 'w')
    pfile.write('\\begin{table}[h]\n')
    pfile.write('\\adjustbox{max width=\\textwidth}{\n')
    pfile.write('\\begin{tabular}{rr||cc||ccc||ccc}\n')
    pfile.write(' & & \multicolumn{2}{c||}{Pixel-wise} & \multicolumn{3}{c||}{Patient-wise} & \multicolumn{3}{c}{Object-wise} \\tabularnewline\n')
    header_line = ''
    for elem in columns[:-1]:
        header_line = header_line + elem + ' & '
    header_line = header_line + columns[-1]
    pfile.write(header_line + '\\tabularnewline\n')
    for index, row in data.iterrows():
        line = str(int(row['Fold'])) + ' & ' + str(int(row['# samples']))
        line = line + ' & ${:05.2f}\pm{:05.2f}$'.format(np.round(row['Dice_mean'] * 100., 2),  np.round(row['Dice_std'] * 100., 2))
        line = line + ' & ${:05.2f}\pm{:05.2f}$'.format(np.round(row['Dice-TP_mean'] * 100., 2),  np.round(row['Dice-TP_std'] * 100., 2))
        for c in range(0, len(df_columns), 1):
            value = row[df_columns[c]]
            line = line + ' & ${:05.2f}$'.format(np.round(value * 100., 2))
        pfile.write(line + '\\tabularnewline\n')
    pfile.write('\\end{tabular}\n')
    pfile.write('}\n')
    pfile.write('\\end{table}')
    pfile.close()


def export_mean_std_df_to_latex(folder, data, suffix=''):
    matrix_filename = os.path.join(folder, 'mean_std_df_latex.txt') if suffix == '' else os.path.join(folder,
                                                                                                       'mean_std_df_' + suffix + '_latex.txt')
    columns = data.columns.values[1:]
    pfile = open(matrix_filename, 'w')
    pfile.write('\\begin{table}[h]\n')
    pfile.write('\\adjustbox{max width=\\textwidth}{\n')
    pfile.write('\\begin{tabular}{l'+('r' * int(len(columns) / 2)) + '}\n')
    pfile.write('\\toprule\n')
    header_line = '{}'
    for elem in columns[::2]:
        header_line = header_line + ' & ' + elem.split('_')[0]
    pfile.write(header_line + '\\tabularnewline\n')
    for index, row in data.iterrows():
        line = ''
        for c in range(0, len(columns), 2):
            mean_value = row[columns[c]]
            std_value = row[columns[c+1]]
            line = line + ' & $' + str(np.round(mean_value * 100., 2)) + '\pm' + str(np.round(std_value * 100., 2)) + '$'
        pfile.write(line + '\\tabularnewline\n')
    pfile.write('\\bottomrule\n')
    pfile.write('\\end{tabular}\n')
    pfile.write('}\n')
    pfile.write('\\end{table}')
    pfile.close()


def export_mean_std_df_to_latex_paper(folder, data, suffix=''):
    matrix_filename = os.path.join(folder, 'mean_std_df_latex_paper.txt') if suffix == '' else os.path.join(folder,
                                                                                                       'mean_std_df_' + suffix + '_latex_paper.txt')
    columns = ['Dice', 'Dice-TP', 'F1-score', 'Recall', 'Precision', 'F1-score', 'Recall', 'Precision']
    df_columns = ['Patient-wise F1', 'Patient-wise recall', 'Patient-wise precision', 'Object-wise F1', 'Object-wise recall', 'Object-wise precision'] #'Fold', '# Samples', 'Dice', 'Dice-TP',
    pfile = open(matrix_filename, 'w')
    pfile.write('\\begin{table}[h]\n')
    pfile.write('\\adjustbox{max width=\\textwidth}{\n')
    pfile.write('\\begin{tabular}{c|cc||ccc||ccc}\n')
    pfile.write(' & \multicolumn{2}{c||}{Pixel-wise} & \multicolumn{3}{c||}{Patient-wise} & \multicolumn{3}{c}{Object-wise} \\tabularnewline\n')
    header_line = ' & '
    for elem in columns[:-1]:
        header_line = header_line + elem + ' & '
    header_line = header_line + columns[-1]
    pfile.write(header_line + '\\tabularnewline\n')
    line = ''
    for index, row in data.iterrows():
        line = line + ' & ${:05.2f}\pm{:05.2f}$'.format(np.round(row['Dice_mean'] * 100., 2),  np.round(row['Dice_std'] * 100., 2))
        line = line + ' & ${:05.2f}\pm{:05.2f}$'.format(np.round(row['Dice-TP_mean'] * 100., 2),  np.round(row['Dice-TP_std'] * 100., 2))
        for c in range(0, len(df_columns), 1):
            line = line + ' & ${:05.2f}\pm{:05.2f}$'.format(np.round(row[df_columns[c] + '_mean'] * 100., 2),
                                                            np.round(row[df_columns[c] + '_std'] * 100., 2))
        pfile.write(line + '\\tabularnewline\n')
    pfile.write('\\end{tabular}\n')
    pfile.write('}\n')
    pfile.write('\\end{table}')
    pfile.close()
