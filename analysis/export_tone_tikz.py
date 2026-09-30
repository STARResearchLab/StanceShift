"""Print pgfplots coordinates of the LIWC Tone histograms used in the paper's tone figures.

Main figure (latex_fig/meme_polarization.tex): Tone of the original / add / meme n_i from outputs/liwc
(rows without Tone dropped, clipped to [0, 100]), 10 bins of width 10, share of rows in %.
Topic figures (latex_fig/meme_polarization_{name}.tex): bin counts from outputs/topics/topic_tone_distribution.csv.

Usage: python analysis/export_tone_tikz.py
"""
import argparse
import os

import numpy as np
import pandas as pd

NAMES = {'claude': 'Claude', 'deepseek': 'DeepSeek', 'llama': 'Llama'}
STRATEGIES = {'Original': 'original', 'Add': 'add', 'Meme': 'meme'}
TOPIC_FIGURES = {0: 'ai_model_evaluation', 1: 'software', 2: 'cost', 3: 'consciousness'}
BIN_EDGES = np.linspace(0, 100, 11)


def coordinates(shares):
    return ' '.join(f'({i * 10 + 5}, {v:.1f})' for i, v in enumerate(shares))


def tone_shares(path):
    tone = pd.to_numeric(pd.read_csv(path)['Tone'], errors='coerce').dropna().clip(lower=0, upper=100)
    counts, _ = np.histogram(tone, bins=BIN_EDGES)
    return counts / len(tone) * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print pgfplots coordinates for the tone figures.')
    parser.add_argument('--liwc_dir', default='outputs/liwc')
    parser.add_argument('--topic_file', default='outputs/topics/topic_tone_distribution.csv')
    args = parser.parse_args()

    print('% latex_fig/meme_polarization.tex')
    for topic, name in NAMES.items():
        for strategy, suffix in STRATEGIES.items():
            shares = tone_shares(os.path.join(args.liwc_dir, f'{topic}_{suffix}.csv'))
            print(f'{name}-{strategy} {coordinates(shares)}')

    dist = pd.read_csv(args.topic_file, index_col=0)
    bins = [c for c in dist.columns if '-' in c]
    for topic, figure in TOPIC_FIGURES.items():
        print(f'\n% latex_fig/meme_polarization_{figure}.tex')
        for strategy in STRATEGIES:
            row = dist[(dist['add_strategy_topic'] == topic) & (dist['strategy'] == strategy)].iloc[0]
            counts = row[bins].astype(float).values
            print(f'{row["topic_label"]}-{strategy} {coordinates(counts / counts.sum() * 100)}')
