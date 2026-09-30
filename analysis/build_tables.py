"""Build the stance tables in results/ from the simulator outputs in outputs/stances.

results/{simulator}_text.csv  conv_id, topic, observed, inferred, paraphrase, explain, add
    One row per conversation whose five responses all contain a stance.
results/{simulator}_meme.csv  conv_id, topic, template_id, inferred, meme
    One row per conversation x template whose meme and inferred responses contain a stance.
results/gpt-5.2-2025-12-11_domains_{text,meme}.csv
    The same tables for the four additional domains of Appendix G (one meme template per conversation).

Usage: python analysis/build_tables.py [--simulators ...]
"""
import argparse
import json
import os
import re

import pandas as pd

TOPICS = ['deepseek', 'claude', 'llama']
DOMAINS = ['nuclear', 'keto', 'ubi', 'spacexipo']
SIMULATORS = ['gpt-5.2-2025-12-11', 'claude-sonnet-4-6', 'qwen3.5-plus-2026-02-15']
TEXT_CONDITIONS = ['observed', 'inferred', 'paraphrase', 'explain', 'add']
STANCE_RE = re.compile(r'"stance"\s*:\s*"([^"]+)"')


def parse_stance(response):
    """Lower-cased 'stance' field of a response (JSON, else regex), or None if there is none."""
    if not isinstance(response, str):
        return None
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.strip())
    try:
        stance = json.loads(text).get('stance')
    except (ValueError, AttributeError):
        match = STANCE_RE.search(text)
        stance = match.group(1) if match else None
    return str(stance).strip().lower() if stance else None


def load_stances(simulator, topic, condition, stance_dir='outputs/stances'):
    """Rows of outputs/stances/{simulator}/{topic}_{condition}.csv with a parsed stance column."""
    df = pd.read_csv(os.path.join(stance_dir, simulator, f'{topic}_{condition}.csv'))
    df['stance'] = df['response'].map(parse_stance)
    return df.dropna(subset=['stance']).drop(columns='response')


def build_text(simulator, stance_dir, topics=TOPICS):
    parts = []
    for topic in topics:
        table = None
        for condition in TEXT_CONDITIONS:
            df = load_stances(simulator, topic, condition, stance_dir).rename(columns={'stance': condition})
            table = df if table is None else table.merge(df, on='conv_id', how='inner')
        table.insert(1, 'topic', topic)
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def build_meme(simulator, stance_dir, topics=TOPICS):
    parts = []
    for topic in topics:
        inferred = load_stances(simulator, topic, 'inferred', stance_dir).rename(columns={'stance': 'inferred'})
        meme = load_stances(simulator, topic, 'meme', stance_dir).rename(columns={'stance': 'meme'})
        table = meme.merge(inferred, on='conv_id', how='inner')
        table.insert(1, 'topic', topic)
        parts.append(table[['conv_id', 'topic', 'template_id', 'inferred', 'meme']])
    return pd.concat(parts, ignore_index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build results/{simulator}_{text,meme}.csv from outputs/stances.')
    parser.add_argument('--simulators', nargs='+', default=SIMULATORS, choices=SIMULATORS)
    parser.add_argument('--stance_dir', default='outputs/stances')
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    jobs = [(simulator, TOPICS, simulator) for simulator in args.simulators]
    if SIMULATORS[0] in args.simulators:  # the additional domains have GPT-5.2 outputs only
        jobs.append((SIMULATORS[0], DOMAINS, f'{SIMULATORS[0]}_domains'))
    for simulator, topics, name in jobs:
        for kind, build in [('text', build_text), ('meme', build_meme)]:
            table = build(simulator, args.stance_dir, topics)
            path = os.path.join(args.output_dir, f'{name}_{kind}.csv')
            table.to_csv(path, index=False)
            counts = '/'.join(str(n) for n in table.groupby('topic', sort=False).size())
            print(f'{path}: {len(table)} rows ({counts} {"/".join(topics)})')
