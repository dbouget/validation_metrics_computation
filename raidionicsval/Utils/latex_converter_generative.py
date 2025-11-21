import os
import numpy as np
import pandas as pd
from typing import List


def sci_notation_latex(x, sig=2):
    """
    Convert a float x to LaTeX scientific notation.
    """
    if x == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exp
    return f'{coeff:.{sig}g}e^{{{exp}}}'

def export_fold_averages_latex(filename, suffix=''):
    fold_average_df = pd.read_csv(filename)
    latex_filename = os.path.join(os.path.dirname(filename), 'fold_averages_latex.txt') if suffix == '' else\
        os.path.join(os.path.dirname(filename), f'fold_averages_{suffix}_latex.txt')
    columns = fold_average_df.columns.values

    unique_metrics = np.unique([x.replace('- Mean', '').replace('- Std', '').strip() for x in list(columns[2:])])
    sw_diff_metrics = [item for item in unique_metrics if "SW" in item and "Flicker" not in item and "Consistency" not in item]
    pw_diff_metrics = [item for item in unique_metrics if "PW" in item and "Flicker" not in item and "Consistency" not in item]
    sw_rel_metrics = [item for item in unique_metrics if
                       "SW" in item and ("Flicker" in item or "Consistency" in item)]
    pw_rel_metrics = [item for item in unique_metrics if
                       "PW" in item and ("Flicker" in item or "Consistency" in item)]
    sw_diff_codes = ''.join(['c'] * len(sw_diff_metrics))
    pw_diff_codes = ''.join(['c'] * len(pw_diff_metrics))
    sw_rel_codes = ''.join(['c'] * len(sw_rel_metrics))
    pw_rel_codes = ''.join(['c'] * len(pw_rel_metrics))
    pfile = open(latex_filename, 'w')
    pfile.write('\\begin{table}[h]\n')
    pfile.write('\\adjustbox{max width=\\textwidth}{\n')
    pfile.write(f'\\begin{{tabular}}{{rr||{sw_diff_codes}||{pw_diff_codes}||}}\n')
    pfile.write(
        f' & & \multicolumn{{{len(sw_diff_codes)}}}{{c||}}{{Slice-wise}} & \multicolumn{{{len(pw_diff_codes)}}}{{c||}}{{Patient-wise}}\\tabularnewline\n')
    header_line = 'Fold & \# Samples'
    for elem in sw_diff_metrics:
        header_line = header_line + ' & ' + elem
    for elem in pw_diff_metrics:
        header_line = header_line + ' & ' + elem
    pfile.write(header_line + '\\tabularnewline\n')
    pfile.write('\hline\n')
    for f in range(len(fold_average_df)):
        line = str(int(fold_average_df['Fold'].values[f])) + ' & ' + str(
            int(fold_average_df['# samples'].values[f]))
        for m in sw_diff_metrics:
            line = line + ' & ${:06.3f}\pm{:06.3f}$'.format(
                np.round(fold_average_df[f'{m} - Mean - Mean'].values[0], 3),
                np.round(fold_average_df[f'{m} - Std - Mean'].values[0], 3))
        for m in pw_diff_metrics:
            line = line + ' & ${:06.3f}\pm{:06.3f}$'.format(
                np.round(fold_average_df[f'{m} - Mean'].values[0], 3),
                np.round(fold_average_df[f'{m} - Std'].values[0], 3))
        pfile.write(line + '\\tabularnewline\n')
    pfile.write('\\end{tabular}\n')
    pfile.write('}\n')
    pfile.write('\\caption{Slice-wise and patient-wise generative metrics averaged over each fold.}\n')
    pfile.write('\\end{table}')

    pfile.write('\n\n\n\n\n')
    pfile.write('\\begin{table}[h]\n')
    pfile.write('\\adjustbox{max width=\\textwidth}{\n')
    pfile.write(f'\\begin{{tabular}}{{rr||{sw_rel_codes}||{pw_rel_codes}||}}\n')
    pfile.write(
        f' & & \multicolumn{{{len(sw_rel_codes)}}}{{c||}}{{Slice-wise}} & \multicolumn{{{len(pw_rel_codes)}}}{{c||}}{{Patient-wise}}\\tabularnewline\n')
    header_line = 'Fold & \# Samples'
    for elem in sw_rel_metrics:
        header_line = header_line + ' & ' + elem
    for elem in pw_rel_metrics:
        header_line = header_line + ' & ' + elem
    pfile.write(header_line + '\\tabularnewline\n')
    pfile.write('\hline\n')
    for f in range(len(fold_average_df)):
        line = str(int(fold_average_df['Fold'].values[f])) + ' & ' + str(
            int(fold_average_df['# samples'].values[f]))
        for m in sw_rel_metrics:
            line = line + ' & ${:06.3f}\pm{:06.3f}$'.format(
                np.round(fold_average_df[f'{m} - Mean - Mean'].values[0], 3),
                np.round(fold_average_df[f'{m} - Std - Mean'].values[0], 3))
        for m in pw_rel_metrics:
            line = line + ' & ${:06.3f}\pm{:06.3f}$'.format(
                np.round(fold_average_df[f'{m} - Mean'].values[0], 3),
                np.round(fold_average_df[f'{m} - Std'].values[0], 3))
        pfile.write(line + '\\tabularnewline\n')
    pfile.write('\\end{tabular}\n')
    pfile.write('}\n')
    pfile.write('\\caption{Slice-wise and patient-wise generative metrics, for the original (Orig) and generated (Gen) volumes averaged over each fold.}\n')
    pfile.write('\\end{table}')
    pfile.close()

    export_fold_averages_latex_flicker(filename, suffix=suffix)


def export_fold_averages_latex_flicker(filename, suffix=''):
    fold_average_df = pd.read_csv(filename)
    latex_filename = os.path.join(os.path.dirname(filename), 'fold_averages_flicker_latex.txt') if suffix == '' else\
        os.path.join(os.path.dirname(filename), f'fold_averages_flicker_{suffix}_latex.txt')
    columns = fold_average_df.columns.values

    unique_metrics = np.unique([x.replace('- Mean', '').replace('- Std', '').strip() for x in list(columns[2:])])
    sw_rel_metrics = [item for item in unique_metrics if
                       "SW" in item and "Flicker" in item ]
    pw_rel_metrics = [item for item in unique_metrics if
                       "PW" in item and "Flicker" in item]
    sw_rel_codes = ''.join(['c'] * len(sw_rel_metrics))
    pw_rel_codes = ''.join(['c'] * len(pw_rel_metrics))
    pfile = open(latex_filename, 'w')
    pfile.write('\\begin{table}[h]\n')
    pfile.write('\\adjustbox{max width=\\textwidth}{\n')
    pfile.write(f'\\begin{{tabular}}{{rr||{sw_rel_codes}||{pw_rel_codes}||}}\n')
    pfile.write(
        f' & & \multicolumn{{{len(sw_rel_codes)}}}{{c||}}{{Slice-wise}} & \multicolumn{{{len(pw_rel_codes)}}}{{c||}}{{Patient-wise}}\\tabularnewline\n')
    header_line = 'Fold & \# Samples'
    for elem in sw_rel_metrics:
        header_line = header_line + ' & ' + elem.replace("SW", '').replace("Flicker", '').strip()
    for elem in pw_rel_metrics:
        header_line = header_line + ' & ' + elem.replace("PW", '').replace("Flicker", '').strip()
    pfile.write(header_line + '\\tabularnewline\n')
    pfile.write('\hline\n')
    for f in range(len(fold_average_df)):
        line = str(int(fold_average_df['Fold'].values[f])) + ' & ' + str(
            int(fold_average_df['# samples'].values[f]))
        for m in sw_rel_metrics:
            line = line + ' & ${:06.3f}\pm{:06.3f}$'.format(
                np.round(fold_average_df[f'{m} - Mean - Mean'].values[0], 3),
                np.round(fold_average_df[f'{m} - Std - Mean'].values[0], 3))
        for m in pw_rel_metrics:
            line = line + ' & ${:06.3f}\pm{:06.3f}$'.format(
                np.round(fold_average_df[f'{m} - Mean'].values[0], 3),
                np.round(fold_average_df[f'{m} - Std'].values[0], 3))
        pfile.write(line + '\\tabularnewline\n')
    pfile.write('\\end{tabular}\n')
    pfile.write('}\n')
    pfile.write('\\caption{Slice-wise and patient-wise flicker metrics averaged over each fold.}\n')
    pfile.write('\\end{table}')

    pfile.write('\n\n\n')
    pfile.write('\\begin{table}[h]\n')
    pfile.write('\\adjustbox{max width=\\textwidth}{\n')
    pfile.write(f'\\begin{{tabular}}{{rr||{sw_rel_codes}||}}\n')
    pfile.write(
        f' & & \multicolumn{{{len(sw_rel_codes)}}}{{c||}}{{Slice-wise}}\\tabularnewline\n')
    header_line = 'Fold & \# Samples'
    for elem in sw_rel_metrics:
        header_line = header_line + ' & ' + elem.replace("SW", '').replace("Flicker", '').strip()
    pfile.write(header_line + '\\tabularnewline\n')
    pfile.write('\hline\n')
    for f in range(len(fold_average_df)):
        line = str(int(fold_average_df['Fold'].values[f])) + ' & ' + str(
            int(fold_average_df['# samples'].values[f]))
        for m in sw_rel_metrics:
            line = line + ' & ${:.2e}\pm{:.2e}$'.format(
                np.round(fold_average_df[f'{m} - Mean - Mean'].values[0], 3),
                np.round(fold_average_df[f'{m} - Std - Mean'].values[0], 3))
        pfile.write(line + '\\tabularnewline\n')
    pfile.write('\\end{tabular}\n')
    pfile.write('}\n')
    pfile.write('\\caption{Slice-wise flicker metrics averaged over each fold.}\n')
    pfile.write('\\end{table}')

    pfile.write('\n\n\n')
    pfile.write('\\begin{table}[h]\n')
    pfile.write('\\adjustbox{max width=\\textwidth}{\n')
    pfile.write(f'\\begin{{tabular}}{{rr||{pw_rel_codes}||}}\n')
    pfile.write(
        f' & & \multicolumn{{{len(pw_rel_codes)}}}{{c||}}{{Patient-wise}}\\tabularnewline\n')
    header_line = 'Fold & \# Samples'
    for elem in pw_rel_metrics:
        header_line = header_line + ' & ' + elem.replace("PW", '').replace("Flicker", '').strip()
    pfile.write(header_line + '\\tabularnewline\n')
    pfile.write('\hline\n')
    for f in range(len(fold_average_df)):
        line = str(int(fold_average_df['Fold'].values[f])) + ' & ' + str(
            int(fold_average_df['# samples'].values[f]))
        for m in pw_rel_metrics:
            line = line + ' & ${}\pm{}$'.format(
                sci_notation_latex(fold_average_df[f'{m} - Mean'].values[0], 3),
                sci_notation_latex(fold_average_df[f'{m} - Std'].values[0], 3))
        pfile.write(line + '\\tabularnewline\n')
    pfile.write('\\end{tabular}\n')
    pfile.write('}\n')
    pfile.write('\\caption{Patient-wise flicker metrics averaged over each fold.}\n')
    pfile.write('\\end{table}')

    pfile.close()