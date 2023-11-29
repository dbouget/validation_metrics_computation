import os
import numpy as np


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


def export_df_to_latex_paper(folder, data, suffix=''):
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
