#!/usr/bin/python3 -u

import argparse
import os.path as op
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from utils_nmf import eprint, validate_file

plt.rcParams.update({'font.size': 12})

cm = LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#4500FF-466CFF-FFFDFD-FBED56-FFF000
    (0.000, (0.271, 0.000, 1.000)),
    (0.250, (0.275, 0.424, 1.000)),
    (0.500, (1.000, 0.992, 0.992)),
    (0.750, (0.984, 0.929, 0.337)),
    (1.000, (1.000, 0.941, 0.000))))

def plot_atlas(atlas_path, pdf_path=None, nan_orig=False):
    validate_file(atlas_path)
    if not atlas_path.endswith('.csv'):
        eprint('Error setting pdf path')
        exit()
    if pdf_path is None:
        pdf_path = atlas_path[:-4] + '.pdf'
    # plt.rcParams.update({'font.size': 20})
    plt.figure()
    # cm = sns.color_palette("coolwarm", as_cmap=True)
    df = pd.read_csv(atlas_path, index_col=0)
    if nan_orig:
        for c in df.columns:
            if 'Unknown' in c:
                df[c] = np.nan
    mask = df.isnull()
    ax = sns.heatmap(data=df, vmax=1, vmin=0, cmap=cm,
                     mask=mask,
                     yticklabels=False)
    ax.set_facecolor('gray')
    ax.set_ylabel('Features')
    ax.set_xlabel('Reference')
    plt.xticks(rotation=90)
    plt.title(op.basename(pdf_path)[:-4])
    plt.tight_layout()
    plt.savefig(pdf_path)
    eprint(f'Dumped to {pdf_path}')
    return pdf_path


def plot_weights(coef_path):
    validate_file(coef_path)
    if not coef_path.endswith('.csv'):
        eprint('Error setting pdf path')
        exit()
    pdf_path = coef_path[:-9] + '.weights.pdf'
    plt.figure()
    df = pd.read_csv(coef_path, index_col=0)
    ax = sns.heatmap(data=df, vmax=1, vmin=0, cmap='Reds')
    ax.set_ylabel('Reference')
    ax.set_xlabel('Samples')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title(op.basename(pdf_path)[:-4])
    plt.tight_layout()
    plt.savefig(pdf_path)
    eprint(f'Dumped to {pdf_path}')
    return pdf_path



def main():
    args = parse_args()

    plot_atlas(args.atlas_path, args.outpath)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('atlas_path', help='Atlas csv to plot')
    parser.add_argument('--outpath', '-o', help='output path (pdf or png)')
    return parser.parse_args()


if __name__ == '__main__':
    main()
