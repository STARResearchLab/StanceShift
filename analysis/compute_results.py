"""Print the numbers reported in the paper from data/, results/ and outputs/.

Run analysis/build_tables.py first if results/ is missing.
Usage: python analysis/compute_results.py
"""
import argparse
import json
import math
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from build_tables import DOMAINS, SIMULATORS, TOPICS, load_stances

NAMES = {'deepseek': 'DeepSeek', 'claude': 'Claude', 'llama': 'Llama', 'nuclear': 'Nuclear energy',
         'keto': 'Keto diet', 'ubi': 'Universal basic income', 'spacexipo': 'SpaceX IPO'}
TABLE_ORDER = ['claude', 'deepseek', 'llama']
SIMULATOR_NAMES = {'gpt-5.2-2025-12-11': 'GPT-5.2', 'claude-sonnet-4-6': 'Sonnet-4.6',
                   'qwen3.5-plus-2026-02-15': 'Qwen3.5-Plus'}
LABELS = ['negative', 'neutral', 'positive']
SCORE = {'negative': -1, 'neutral': 0, 'positive': 1}
ABLATIONS = [('r_meme', 'meme'), ('r_white_meme', 'white_meme'), ('r_humor', 'humor'),
             ('r_caption_cut', 'caption_cut'), ('r_caption', 'caption')]


def dataset_statistics(skeleton_dir):
    sizes, subreddits, posts, targets, authors = {}, set(), set(), set(), set()
    for topic in TOPICS:
        with open(os.path.join(skeleton_dir, f'{topic}.jsonl')) as f:
            records = [json.loads(line) for line in f]
        sizes[topic] = len(records)
        for record in records:
            subreddits.add(record['subreddit'])
            posts.add(record['post_id'])
            targets.add(record['target_user'])
            node = record['conv']
            while node is not None:
                authors.add(node['author'])  # deleted accounts (null) count as one author
                node = node['child']
    print('Dataset statistics')
    print(f'  conversations {sum(sizes.values()):,} ({", ".join(f"{NAMES[t]} {n}" for t, n in sizes.items())})')
    print(f'  subreddits {len(subreddits)}, posts {len(posts)}, target users {len(targets)}, authors {len(authors):,}')


def table_original_stance(text):
    text = text[text['observed'].isin(LABELS)]
    print(f'\nTable tab:original_stance_performance: GPT-5.2 inferred vs observed stance, N = {len(text):,}')
    print(f'{"Target":<10}{"Accuracy":>10}{"Macro F1":>10}{"Weighted F1":>13}')
    for name, df in [(NAMES[t], text[text['topic'] == t]) for t in TOPICS] + [('Overall', text)]:
        y, p = df['observed'], df['inferred']
        macro = f1_score(y, p, labels=LABELS, average='macro')
        weighted = f1_score(y, p, labels=LABELS, average='weighted')
        print(f'{name:<10}{100 * accuracy_score(y, p):>10.2f}{100 * macro:>10.2f}{100 * weighted:>13.2f}')


def fmt_delta(delta, base):
    value = '0.000' if round(delta, 3) == 0 else f'{delta:+.3f}'
    return f'{value} ({100 * delta / base:+.1f}%)'


def delta_row(name, text, meme):
    """Mean score shift vs the inferred stance; meme: mean over all conversation x template rows."""
    inferred = text['inferred'].map(SCORE)
    base = inferred.mean()
    cells = [fmt_delta((text[s].map(SCORE) - inferred).mean(), base) for s in ['paraphrase', 'explain', 'add']]
    cells.append(fmt_delta(meme['meme'].map(SCORE).mean() - base, base))
    print(f'{name:<10}{base:>9.3f}' + ''.join(f'{c:>19}' for c in cells))


def table_delta(tables, simulators, title, average=False):
    print(f'\n{title}')
    print(f'{"Target":<10}{"Inferred":>9}' + ''.join(f'{s:>19}' for s in ['paraphrase', 'explain', 'add', 'meme']))
    for simulator in simulators:
        text, meme = tables[simulator]
        if len(simulators) > 1:
            print(f'-- {SIMULATOR_NAMES[simulator]}-Inferred Stance')
        for topic in TABLE_ORDER:
            delta_row(NAMES[topic], text[text['topic'] == topic], meme[meme['topic'] == topic])
        if average:
            delta_row('Average', text, meme)


def meme_neutral_to_positive(meme):
    """Mean over targets of #(neutral -> positive) / #(inferred negative or neutral)."""
    rates = []
    for topic in TOPICS:
        df = meme[(meme['topic'] == topic) & meme['inferred'].isin(['negative', 'neutral'])]
        rates.append(100 * ((df['inferred'] == 'neutral') & (df['meme'] == 'positive')).mean())
    print(f'\nMeme neutral -> positive transition rate, mean over targets (GPT-5.2): {sum(rates) / len(rates):.1f}%')


def upward_rate(df):
    """Share (%) of rows with inferred stance negative/neutral whose revised stance is higher."""
    df = df[df['inferred'].isin(['negative', 'neutral'])]
    return 100 * (df['stance'].map(SCORE) > df['inferred'].map(SCORE)).mean()


def table_meme_ablation(simulator, stance_dir, ablation_dir):
    print('\nTable tab:meme-ablation: upward transition rate on the 50-conversation subsets (GPT-5.2)')
    print(f'{"Strategy":<15}' + ''.join(f'{n:>10}' for n in [NAMES[t] for t in TABLE_ORDER] + ['Average']))
    for name, condition in ABLATIONS:
        rates = []
        for topic in TABLE_ORDER:
            with open(os.path.join(ablation_dir, f'{topic}.txt')) as f:
                ids = set(f.read().split())
            inferred = load_stances(simulator, topic, 'inferred', stance_dir).rename(columns={'stance': 'inferred'})
            df = load_stances(simulator, topic, condition, stance_dir).merge(inferred, on='conv_id')
            rates.append(upward_rate(df[df['conv_id'].isin(ids)]))
        print(f'{name:<15}' + ''.join(f'{r:>9.1f}%' for r in rates + [sum(rates) / len(rates)]))


def table_tone_shift(liwc_dir, topics=TOPICS, label='tab:meme-antipolar', column='Model', width=10):
    print(f'\nTable {label}: LIWC Tone of the revised vs the original n_i (rows without Tone dropped)')
    print(f'{column:<{width}}{"Revision":<10}{"Low-to-High":>13}{"High-to-Low":>13}{"Overall":>9}')
    for topic in topics:
        original = pd.read_csv(os.path.join(liwc_dir, f'{topic}_original.csv'))[['conv_id', 'Tone']]
        for strategy in ['add', 'meme']:
            revised = pd.read_csv(os.path.join(liwc_dir, f'{topic}_{strategy}.csv'))[['conv_id', 'Tone']]
            df = revised.merge(original, on='conv_id', suffixes=('', '_original'))
            df = df.dropna(subset=['Tone', 'Tone_original'])
            low = df['Tone_original'] <= 50
            up = (df['Tone'] > df['Tone_original'])[low]
            down = (df['Tone'] < df['Tone_original'])[~low]
            overall = 100 * (up.sum() + down.sum()) / len(df)
            print(f'{NAMES[topic]:<{width}}{strategy.capitalize():<10}{100 * up.mean():>13.1f}'
                  f'{100 * down.mean():>13.1f}{overall:>9.1f}')


def sign_test(diff):
    """Two-sided exact binomial p-value of the signs of diff (zeros and NaN dropped)."""
    pos, neg = int((diff > 0).sum()), int((diff < 0).sum())
    return min(1.0, 2 * sum(math.comb(pos + neg, k) for k in range(min(pos, neg) + 1)) / 2 ** (pos + neg))


def fmt_shift(delta, relative, p):
    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    value = f'{delta:+.3f} ({relative:+.1f}%)'
    return f'{value:>19}{stars:<3}'


def table_domains_delta(text, meme):
    """Mean per-conversation score shift vs the inferred stance (meme: conversations with a meme) and its ratio to
    |mean inferred score|. Stars: sign test of each strategy's shift against the paraphrase shift on the same
    conversations. Average: mean of the per-target values, sign test over all conversations."""
    print('\nTable tab:app-domains-delta: average directional stance shift Delta (Delta / |Inferred|), GPT-5.2, '
          'sign test vs paraphrase (* p<0.05, ** p<0.01, *** p<0.001)')
    strategies = ['paraphrase', 'explain', 'add', 'meme']
    print(f'{"Target":<24}{"Inferred":>9}' + ''.join(f'{s:>19}   ' for s in strategies).rstrip())
    df = text.merge(meme[['conv_id', 'meme']], on='conv_id', how='left')
    shift = pd.DataFrame({s: df[s].map(SCORE) - df['inferred'].map(SCORE) for s in strategies})
    per_target = []
    for topic in DOMAINS:
        rows = df['topic'] == topic
        base = df.loc[rows, 'inferred'].map(SCORE).mean()
        deltas = [shift.loc[rows, s].mean() for s in strategies]
        per_target.append([(d, 100 * d / abs(base)) for d in deltas])
        tests = [sign_test(shift.loc[rows, s] - shift.loc[rows, 'paraphrase']) for s in strategies]
        cells = ''.join(fmt_shift(d, r, p) for (d, r), p in zip(per_target[-1], tests))
        print(f'{NAMES[topic]:<24}{base:>+9.3f}{cells}'.rstrip())
    means = [[sum(v) / len(v) for v in zip(*column)] for column in zip(*per_target)]
    tests = [sign_test(shift[s] - shift['paraphrase']) for s in strategies]
    cells = ''.join(fmt_shift(d, r, p) for (d, r), p in zip(means, tests))
    print(f'{"Average":<24}{"--":>9}{cells}'.rstrip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print the paper numbers.')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--stance_dir', default='outputs/stances')
    parser.add_argument('--liwc_dir', default='outputs/liwc')
    parser.add_argument('--skeleton_dir', default='data/skeletons')
    parser.add_argument('--ablation_dir', default='data/ablation_ids')
    args = parser.parse_args()

    tables = {s: (pd.read_csv(os.path.join(args.results_dir, f'{s}_text.csv')),
                  pd.read_csv(os.path.join(args.results_dir, f'{s}_meme.csv'))) for s in SIMULATORS}
    gpt = SIMULATORS[0]
    dataset_statistics(args.skeleton_dir)
    table_original_stance(tables[gpt][0])
    title = 'average directional stance shift Delta (Delta / Inferred)'
    table_delta(tables, [gpt], f'Table tab:main-delta: {title}, GPT-5.2', average=True)
    table_delta(tables, SIMULATORS, f'Table tab:app-main-delta: {title}, three simulators')
    meme_neutral_to_positive(tables[gpt][1])
    table_meme_ablation(gpt, args.stance_dir, args.ablation_dir)
    table_tone_shift(args.liwc_dir)
    table_domains_delta(pd.read_csv(os.path.join(args.results_dir, f'{gpt}_domains_text.csv')),
                        pd.read_csv(os.path.join(args.results_dir, f'{gpt}_domains_meme.csv')))
    table_tone_shift(args.liwc_dir, DOMAINS, 'tab:app-domains-tone', 'Topic', 24)
