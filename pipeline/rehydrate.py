"""Fetch the comment texts from Reddit and assemble the full conversations.

data/skeletons/{topic}.jsonl holds every conversation without its texts: each message has its comment_id, author and
the SHA-256 of the text used in the paper (text_sha256). This script fetches the texts with the Reddit API (PRAW,
read-only) and writes data/conversations/{topic}.jsonl with the message key text instead of text_sha256. Comments
that were edited, deleted or removed after the crawl come back as Reddit shows them now; a comment that the API no
longer returns gets the text '[deleted]'. data/conversations/rehydration_report.json counts per topic the messages
whose text equals the paper version (same SHA-256), changed, deleted or removed, and missing messages, and lists the
conversations that contain messages of the last three kinds.

Credentials: create an app of type "script" at https://www.reddit.com/prefs/apps and set REDDIT_CLIENT_ID,
REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT (e.g. 'stanceshift-rehydrate by u/<your username>').
Usage: python pipeline/rehydrate.py [-t deepseek claude ...]
"""
import argparse
import hashlib
import json
import os

TOPICS = ['deepseek', 'claude', 'llama', 'nuclear', 'keto', 'ubi', 'spacexipo']
RECORD_KEYS = ['conv_id', 'subreddit', 'post_id', 'target_user', 'conv']
CREDENTIALS = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']
REMOVED = {'[deleted]', '[removed]', '[ Removed by Reddit ]'}
STATUSES = ['identical', 'changed', 'deleted_or_removed', 'missing']
BATCH = 100


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_nodes(record):
    node = record['conv']
    while node is not None:
        yield node
        node = node['child']


def sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_cache(path):
    """comment_id -> body, None for a comment the API did not return; later lines win."""
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    item = json.loads(line)
                except ValueError:  # a line cut off by an interrupted run
                    continue
                cache[item['id']] = item['body']
    return cache


def reddit_client():
    missing = [name for name in CREDENTIALS if not os.environ.get(name)]
    if missing:
        raise SystemExit(f'Please set {", ".join(missing)}. Create an app of type "script" at '
                         'https://www.reddit.com/prefs/apps and use its client id and secret.')
    try:
        import praw
    except ImportError:
        raise SystemExit('PRAW is required: pip install praw')
    reddit = praw.Reddit(client_id=os.environ['REDDIT_CLIENT_ID'], client_secret=os.environ['REDDIT_CLIENT_SECRET'],
                         user_agent=os.environ['REDDIT_USER_AGENT'], check_for_updates=False)
    reddit.read_only = True
    return reddit


def fetch(comment_ids, cache_path):
    """Fetch the comments in batches of 100 and append their bodies to the cache after every batch."""
    reddit = reddit_client()
    import prawcore
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    fetched = {}
    # A killed run can leave a cut-off last line; start a new line so the next record stays readable.
    if os.path.exists(cache_path) and os.path.getsize(cache_path):
        with open(cache_path, 'rb') as f:
            f.seek(-1, os.SEEK_END)
            cut = f.read(1) != b'\n'
        if cut:
            with open(cache_path, 'a') as f:
                f.write('\n')
    with open(cache_path, 'a') as f:
        for i in range(0, len(comment_ids), BATCH):
            batch = comment_ids[i:i + BATCH]
            try:
                bodies = {c.id: c.body for c in reddit.info(fullnames=['t1_' + c for c in batch])}
            except prawcore.exceptions.PrawcoreException as e:
                raise SystemExit(f'Reddit API error: {e!r}. The comments fetched so far are cached in {cache_path}; rerun to resume.')
            fetched.update((c, bodies.get(c)) for c in batch)
            f.write(''.join(json.dumps({'id': c, 'body': fetched[c]}) + '\n' for c in batch))
            f.flush()
            print(f'fetched {i + len(batch):,}/{len(comment_ids):,} comments')
    return fetched


def rehydrate(record, cache):
    """The full conversation of a skeleton record and the status of each message."""
    nodes, statuses = [], []
    for node in iter_nodes(record):
        body = cache.get(node['comment_id'])
        text = '[deleted]' if body is None else body
        if sha256(text) == node['text_sha256']:
            statuses.append('identical')
        else:
            statuses.append('missing' if body is None else 'deleted_or_removed' if body in REMOVED else 'changed')
        nodes.append({'comment_id': node['comment_id'], 'author': node['author'], 'text': text})
    child = None
    for node in reversed(nodes):
        node['child'] = child
        child = node
    return {k: child if k == 'conv' else record[k] for k in RECORD_KEYS}, statuses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch the comment texts of the released conversations from Reddit '
                                                 '(credentials from REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT).')
    parser.add_argument('-t', '--topic', nargs='+', default=TOPICS, choices=TOPICS, metavar='TOPIC',
                        help=f'One or more of {", ".join(TOPICS)}. Default: all.')
    parser.add_argument('--skeleton_dir', default='data/skeletons')
    parser.add_argument('--output_dir', default='data/conversations')
    parser.add_argument('--cache', default=None,
                        help='Fetched comment bodies, reused on reruns. Default: {output_dir}/fetched_comments.jsonl')
    parser.add_argument('--retry_missing', action='store_true', help='Fetch again the comments the API did not return before.')
    args = parser.parse_args()

    topics = [t for t in TOPICS if t in args.topic]
    skeletons = {t: read_jsonl(f'{args.skeleton_dir}/{t}.jsonl') for t in topics}
    comment_ids = list(dict.fromkeys(node['comment_id'] for t in topics for r in skeletons[t] for node in iter_nodes(r)))
    cache_path = args.cache or f'{args.output_dir}/fetched_comments.jsonl'
    cache = load_cache(cache_path)
    todo = [c for c in comment_ids if c not in cache or (args.retry_missing and cache[c] is None)]
    print(f'{len(comment_ids):,} comments, {len(comment_ids) - len(todo):,} cached, {len(todo):,} to fetch')
    if todo:
        cache.update(fetch(todo, cache_path))

    report_path = f'{args.output_dir}/rehydration_report.json'
    report = {}
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
    os.makedirs(args.output_dir, exist_ok=True)
    for t in topics:
        counts, affected, n_identical = dict.fromkeys(STATUSES, 0), {s: [] for s in STATUSES[1:]}, 0
        with open(f'{args.output_dir}/{t}.jsonl', 'w') as f:
            for skeleton in skeletons[t]:
                record, statuses = rehydrate(skeleton, cache)
                f.write(json.dumps(record) + '\n')
                n_identical += set(statuses) == {'identical'}
                for s in STATUSES:
                    counts[s] += statuses.count(s)
                    if s in affected and s in statuses:
                        affected[s].append(record['conv_id'])
        report[t] = {'conversations': len(skeletons[t]), 'conversations_identical': n_identical,
                     'messages': sum(counts.values()), **counts, 'conv_ids': affected}
        print(f'{t}: {len(skeletons[t])} conversations ({n_identical} identical to the paper version), '
              f'{sum(counts.values()):,} messages: {counts["identical"]:,} identical, {counts["changed"]:,} changed, '
              f'{counts["deleted_or_removed"]:,} deleted/removed, {counts["missing"]:,} missing -> {args.output_dir}/{t}.jsonl')
    with open(report_path, 'w') as f:
        json.dump({t: report[t] for t in TOPICS if t in report}, f, indent=1)
    print(f'Wrote {report_path}')
