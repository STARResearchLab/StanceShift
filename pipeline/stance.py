import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from llm import SIMULATOR_MODELS, simulate_stance, stance_client

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'analysis'))
from build_tables import parse_stance  # the stance parser of the result tables

TOPICS = ['deepseek', 'claude', 'llama', 'nuclear', 'keto', 'ubi', 'spacexipo']
CONDITIONS = ['observed', 'inferred', 'paraphrase', 'explain', 'add', 'meme', 'humor', 'caption', 'caption_cut']


def natural_key(conv_id):
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', str(conv_id))]


def pending(records, done_rows):
    """Records that no done row covers. A row is identified by (conv_id, template_id, k), k counting the earlier rows
    with the same conv_id and template_id; done rows without template_id cover the records of their conversation."""
    left = Counter((r['conv_id'], r.get('template_id')) for r in done_rows)
    todo = []
    for r in records:
        key = next((k for k in [(r['conv_id'], r.get('template_id')), (r['conv_id'], None)] if left[k] > 0), None)
        if key is None:
            todo.append(r)
        else:
            left[key] -= 1
    return todo


def read_scored(path):
    """Rows of an existing output file whose response has a stance."""
    rows = []
    for row in pd.read_csv(path).to_dict('records'):
        if parse_stance(row['response']) is not None:
            kept = {'conv_id': row['conv_id'], 'response': row['response']}
            if not pd.isna(row.get('template_id')):
                kept['template_id'] = int(row['template_id'])
            rows.append(kept)
    return rows


def save_responses(path, rows):
    rows = sorted(rows, key=lambda r: (natural_key(r['conv_id']), -1 if r.get('template_id') is None else r['template_id']))
    columns = ['conv_id', 'template_id', 'response'] if any('template_id' in r for r in rows) else ['conv_id', 'response']
    df = pd.DataFrame(rows, columns=columns)
    if 'template_id' in columns:
        df['template_id'] = df['template_id'].astype('Int64')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    df.to_csv(path, index=False)


def score_one(record, args):
    response = simulate_stance(args.model, record['prompt_text'], img=record.get('prompt_img'), compressed=args.compressed)
    row = {'conv_id': record['conv_id'], 'response': response}
    if record.get('template_id') is not None:
        row['template_id'] = record['template_id']
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query the stance simulator f on stance prompts.')
    parser.add_argument('-t', '--topic', required=True, choices=TOPICS)
    parser.add_argument('-s', '--condition', required=True, choices=CONDITIONS)
    parser.add_argument('--model', default=SIMULATOR_MODELS[0], choices=SIMULATOR_MODELS, help='Stance simulator.')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--sample_num', type=int, default=-1,
                        help='Score only the first N prompts; the default output file then gets the suffix _n{N}.')
    parser.add_argument('--resume', action='store_true',
                        help='Keep rows of the existing output file whose response has a stance label and score the rest.')
    parser.add_argument('--compressed', action='store_true', help='Send meme images as in-memory JPEGs of about 30-50 KB.')
    parser.add_argument('--input_file', default=None, help='Default: outputs/prompts/stance/{topic}_{condition}.jsonl')
    parser.add_argument('--output_file', default=None, help='Default: outputs/stances/{model}/{topic}_{condition}.csv')
    args = parser.parse_args()
    if args.num_workers < 1:
        raise ValueError('--num_workers must be at least 1.')
    stance_client(args.model)  # stop here if the package or an API key is missing

    input_file = args.input_file or f'outputs/prompts/stance/{args.topic}_{args.condition}.jsonl'
    suffix = f'_n{args.sample_num}' if args.sample_num > 0 else ''
    output_file = args.output_file or f'outputs/stances/{args.model}/{args.topic}_{args.condition}{suffix}.csv'

    with open(input_file) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.sample_num > 0:
        records = records[:args.sample_num]
    if not records:
        raise SystemExit(f'{input_file} has no prompts; {output_file} was not written.')

    missing = [r['prompt_img'] for r in records if r.get('prompt_img') and not os.path.isfile(r['prompt_img'])]
    if missing:
        raise SystemExit(f'{len(missing)} prompt images are missing, e.g. {missing[0]}. Render the memes with revise.py first.')

    results = []
    if args.resume and os.path.exists(output_file):
        results = read_scored(output_file)
        records = pending(records, results)
        print(f'[resume] {len(results)} rows already scored in {output_file}; {len(records)} remaining')

    scored = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(score_one, record, args) for record in records]
        try:
            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures)), start=1):
                row = future.result()
                results.append(row)
                scored += parse_stance(row['response']) is not None
                if i % 10 == 0 and (scored or args.resume):
                    save_responses(output_file, results)
        except BaseException:
            executor.shutdown(cancel_futures=True)  # an error that a retry cannot fix stops the queued prompts
            raise

    if not (scored or args.resume):
        raise SystemExit(f'No response has a stance; {output_file} was not written.')
    save_responses(output_file, results)
    errors = sum(parse_stance(r['response']) is None for r in results)
    print(f'Wrote {len(results)} rows to {output_file} ({errors} without a stance)')
