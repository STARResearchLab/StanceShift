import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from llm import IMAGE_MODEL, REVISION_MODELS, edit_image_with_gpt, image_client, revise_text, revision_client

TOPICS = ['deepseek', 'claude', 'llama', 'nuclear', 'keto', 'ubi', 'spacexipo']
STRATEGIES = ['paraphrase', 'explain', 'add', 'meme', 'humor', 'caption', 'caption_cut']


def natural_key(conv_id):
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', str(conv_id))]


def sort_key(row):
    template_id = row.get('template_id')
    return (natural_key(row['conv_id']), -1 if template_id is None else template_id)


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


def meme_path(image_dir, conv_id, template_id):
    return f'{image_dir}/{conv_id}_{template_id}.png'


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        for row in sorted(rows, key=sort_key):
            f.write(json.dumps(row) + '\n')


def has_text(row, field):
    text = row.get(field)
    return bool(text and text.strip())


def is_complete(row, args):
    if args.condition != 'meme':
        return has_text(row, 'revised_message')
    return has_text(row, 'meme_text') and os.path.exists(meme_path(args.image_dir, row['conv_id'], row.get('template_id')))


def revise_one(request, args, meme_text=None):
    """Returns the revision row and whether its meme image is missing. A given meme_text is only rendered."""
    row = {'conv_id': request['conv_id']}
    if 'template_id' in request:
        row['template_id'] = request['template_id']
    if args.condition != 'meme':
        row['revised_message'] = revise_text(request['prompt_text'], args.revision_model, temperature=args.temperature)
        return row, False

    template_path = f"meme_templates/{request['template_id']}.jpg"
    if meme_text is None:
        meme_text = revise_text(request['prompt_text'], args.revision_model, temperature=args.temperature, image_path=template_path)
    row['meme_text'] = meme_text
    output_path = meme_path(args.image_dir, request['conv_id'], request['template_id'])
    rendered = edit_image_with_gpt(meme_text, template_path, output_path, model=args.image_model)
    return row, rendered is None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Revise the last other-user message n_i with the revision model g.')
    parser.add_argument('-t', '--topic', required=True, choices=TOPICS)
    parser.add_argument('-s', '--condition', required=True, choices=STRATEGIES, help='Revision strategy.')
    parser.add_argument('--revision_model', default=REVISION_MODELS[0], choices=REVISION_MODELS)
    parser.add_argument('--image_model', default=IMAGE_MODEL, help='Image model that renders the meme text onto the template (meme only).')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--sample_num', type=int, default=-1,
                        help='Revise only the first N requests; the default output file and meme directory then get the suffix _n{N}.')
    parser.add_argument('--resume', action='store_true',
                        help='Keep completed rows of the existing output file (for meme: text present and image rendered) and '
                             'revise the rest; meme rows that have text but no image only get the image rendered.')
    parser.add_argument('--input_file', default=None, help='Default: outputs/prompts/revision/{topic}_{condition}.jsonl')
    parser.add_argument('--output_file', default=None, help='Default: outputs/revisions/{topic}_{condition}.jsonl')
    parser.add_argument('--meme_dir', default='outputs/memes',
                        help='Memes are written to {meme_dir}/{topic}/{conv_id}_{template_id}.png '
                             '({meme_dir}_n{N}/{topic}/ with --sample_num N).')
    args = parser.parse_args()
    if args.num_workers < 1:
        raise ValueError('--num_workers must be at least 1.')
    revision_client(args.revision_model)  # stop here if a package or an API key is missing
    if args.condition == 'meme':
        image_client()

    input_file = args.input_file or f'outputs/prompts/revision/{args.topic}_{args.condition}.jsonl'
    suffix = f'_n{args.sample_num}' if args.sample_num > 0 else ''
    output_file = args.output_file or f'outputs/revisions/{args.topic}_{args.condition}{suffix}.jsonl'
    args.image_dir = f'{args.meme_dir}{suffix}/{args.topic}'

    with open(input_file) as f:
        requests = [json.loads(line) for line in f if line.strip()]
    if args.sample_num > 0:
        requests = requests[:args.sample_num]
    if not requests:
        raise SystemExit(f'{input_file} has no requests; {output_file} was not written.')

    results, meme_texts = [], {}
    if args.resume and os.path.exists(output_file):
        with open(output_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        results = [row for row in rows if is_complete(row, args)]
        meme_texts = {(row['conv_id'], row.get('template_id')): row['meme_text'] for row in rows if has_text(row, 'meme_text')}
        requests = pending(requests, results)
        image_only = sum((r['conv_id'], r.get('template_id')) in meme_texts for r in requests)
        print(f'[resume] {len(results)} completed rows in {output_file}; {len(requests)} remaining'
              + (f' ({image_only} with meme text, image only)' if image_only else ''))

    failed, unrendered = 0, 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(revise_one, r, args, meme_texts.get((r['conv_id'], r.get('template_id')))): r
                   for r in requests}
        try:
            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures)), start=1):
                request = futures[future]
                try:
                    row, image_missing = future.result()
                except RuntimeError as e:  # the revision model failed on every attempt
                    failed += 1
                    print(f"[error] {request['conv_id']} {request.get('template_id', '')}: {e}")
                    continue
                results.append(row)
                unrendered += image_missing
                if i % 10 == 0:
                    write_jsonl(output_file, results)
        except BaseException:
            executor.shutdown(cancel_futures=True)  # an error that a retry cannot fix stops the queued requests
            raise

    if not results:
        raise SystemExit(f'No request succeeded; {output_file} was not written.')
    write_jsonl(output_file, results)
    print(f'Wrote {len(results)} rows to {output_file}')
    if unrendered:
        print(f'{unrendered} memes could not be rendered; rerun with --resume to render them.')
    if failed:
        print(f'{failed} requests failed and were not written; rerun with --resume to retry them.')
