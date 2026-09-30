import argparse
import json
import math
import os

import pandas as pd
from tqdm import tqdm

TOPICS = ['deepseek', 'claude', 'llama']
RECORD_KEYS = ['conv_id', 'subreddit', 'post_id', 'target_user', 'conv']


def find_top_comment(df_post, commentid):
    # Walk from a comment up to its top-level comment; returns the top-level id and the path 'c_n#...#c_0#post_id'.
    flow = commentid
    tem_df = df_post[df_post['CommentID'] == commentid]
    current_depth = tem_df['Depth'].values[0]

    if current_depth == 0:
        parentid = tem_df['ParentID'].values[0].replace('t3_', '')
        return commentid, f'{flow}#{parentid}'
    parentid = tem_df['ParentID'].values[0]
    flow = f'{flow}#{parentid}'
    while current_depth != 0:
        commentid = parentid.replace('t1_', '')
        current_depth = df_post[df_post['CommentID'] == commentid]['Depth'].values[0]
        parentid = df_post[df_post['CommentID'] == commentid]['ParentID'].values[0]
        flow = f'{flow}#{parentid}'
    return commentid, flow.replace('t1_', '').replace('t3_', '')


def remove_substrings(strings):
    return [s for s in strings if not any(s != other and s in other for other in strings)]


def build_comment_tree(comments):
    # comments run from the target's comment up to the top-level comment; the result is nested top-down.
    tree = None
    for comment in comments:
        tree = {'comment_id': comment['comment_id'], 'author': comment['author'], 'text': comment['text'], 'child': tree}
    return tree


def get_author_conversation_flow(df, post_id, author_id):
    df_post = df[df['PostID'] == post_id].copy()
    if df_post.empty:
        return []
    subreddit = df_post['Subreddit'].values[0]
    depths = list(df_post[df_post['Author'] == author_id]['Depth'].unique())

    flow_list = []
    for depth in sorted(depths, reverse=True):
        if depth != -1:
            commentid_list = df_post[(df_post['Author'] == author_id) & (df_post['Depth'] == depth)]['CommentID'].tolist()
            for commentid in commentid_list:
                _, flow = find_top_comment(df_post, commentid)
                flow_list.append(flow)
    flow_list = remove_substrings(flow_list)

    conv = []
    for flow in flow_list:
        flst = flow.split('#')
        if len(flst) > 2:
            comments = []
            for commentid in flst[:-1]:
                row = df_post[df_post['CommentID'] == commentid]
                comments.append({'comment_id': commentid, 'text': row['Comment'].values[0], 'author': row['Author'].values[0]})
            conv.append({'subreddit': subreddit, 'post_id': post_id, 'target_user': author_id, 'conv': build_comment_tree(comments)})
    return conv


def iter_nodes(record):
    node = record['conv']
    while node is not None:
        yield node
        node = node['child']


def is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def has_removed_message(record):
    # A message whose text is missing, '[deleted]' or '[removed]', or contains 'Redact'.
    for node in iter_nodes(record):
        text = node['text']
        if is_missing(text) or str(text) in ['[deleted]', '[removed]'] or 'Redact' in str(text):
            return True
    return False


def target_mentions_keyword(record, keyword):
    return any(node['author'] == record['target_user'] and keyword in str(node['text']).lower() for node in iter_nodes(record))


def filter_conversations(records, keyword, min_messages=3):
    """Filters that produce the paper set. Returns the kept records and the count after each step."""
    counts = [('extracted', len(records))]
    records = [r for r in records if not has_removed_message(r)]
    counts.append(('no deleted/removed/redacted message', len(records)))
    records = [r for r in records if target_mentions_keyword(r, keyword)]
    counts.append((f'target user mentions "{keyword}"', len(records)))
    records = [r for r in records if sum(1 for _ in iter_nodes(r)) >= min_messages]
    counts.append((f'at least {min_messages} messages', len(records)))
    return records, counts


def extract_conversations(df_comments, topic, comment_count):
    # Every author with at least comment_count comments under a post is a target user.
    posts_to_look_into = {}
    n_targets = 0
    for post_id in df_comments['PostID'].unique():
        counts = df_comments[df_comments['PostID'] == post_id]['Author'].value_counts()
        target_authors = [a for a in counts[counts >= comment_count].index.tolist() if a not in ('[deleted]', 'AutoModerator')]
        if target_authors:
            posts_to_look_into[post_id] = target_authors
            n_targets += len(target_authors)

    records = []
    for post_id in tqdm(posts_to_look_into):
        for author in posts_to_look_into[post_id]:
            records += get_author_conversation_flow(df_comments, post_id, author)
    print(f'{topic}: {n_targets} target users, {len(records)} conversations')

    # conv_id is the position in the extracted list and is kept through the filters.
    for idx, record in enumerate(records):
        record['conv_id'] = f'{topic}_{idx}'
        for node in iter_nodes(record):
            if is_missing(node['author']):
                node['author'] = None
    return [{k: record[k] for k in RECORD_KEYS} for record in records]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract target-user conversations from a raw Reddit comment dump.')
    parser.add_argument('-t', '--topic', nargs='+', required=True, choices=TOPICS)
    parser.add_argument('-cc', '--comment_count', type=int, default=5,
                        help='Minimum number of comments by the target user under one post.')
    parser.add_argument('--input_dir', default='data/raw',
                        help='Directory with the raw comment dumps {topic}.csv (columns Subreddit, PostID, Post Title, Post Body, '
                             'CommentID, Comment, Created_UTC, Author, Depth, ParentID, Score, uid).')
    parser.add_argument('--output_dir', default='data/conversations')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for topic in [t for t in TOPICS if t in args.topic]:
        df_comments = pd.read_csv(f'{args.input_dir}/{topic}.csv')
        records = extract_conversations(df_comments, topic, args.comment_count)
        records, counts = filter_conversations(records, keyword=topic)
        print(' -> '.join(f'{name}: {n}' for name, n in counts))
        with open(f'{args.output_dir}/{topic}.jsonl', 'w') as f:
            for record in records:
                f.write(json.dumps({k: record[k] for k in RECORD_KEYS}) + '\n')
        print(f'Wrote {len(records)} conversations to {args.output_dir}/{topic}.jsonl')
