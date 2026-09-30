import argparse
import json
import os
from collections import defaultdict

TOPICS = ['deepseek', 'claude', 'llama', 'nuclear', 'keto', 'ubi', 'spacexipo']
TARGET_NAMES = {'deepseek': 'DeepSeek', 'claude': 'Claude', 'llama': 'Llama', 'nuclear': 'nuclear energy',
                'keto': 'the keto diet', 'ubi': 'universal basic income', 'spacexipo': 'the SpaceX IPO'}
TEMPLATE_IDS = [0, 1, 3, 4, 9]
STANCE_CONDITIONS = ['observed', 'inferred']
TEXT_STRATEGIES = ['paraphrase', 'explain', 'add']
ABLATION_STRATEGIES = ['humor', 'caption', 'caption_cut']
STRATEGIES = TEXT_STRATEGIES + ['meme'] + ABLATION_STRATEGIES
CONDITIONS = STANCE_CONDITIONS + STRATEGIES
CAPTION_FILES = {'caption': 'meme_templates/captions.jsonl', 'caption_cut': 'meme_templates/captions_visual.jsonl'}
IMAGE_PLACEHOLDER = '<Image Here>'


# Trailing spaces inside the stance prompts are written as \x20.
def observed_prompt(conv_transcript, target_model):
    return f"""
Instruction:\x20
You will be given a multi-party conversation that includes messages from a target user and other users. Your task is to determine the target user’s stance toward {target_model}.

Input:
* A conversation transcript involving multiple users
* The target user is explicitly identified as: [TARGET USER]
* Other users may express opinions, but your output must be based only on the target user’s own statements, not others.

Your task:
* Read the conversation.
* Focus exclusively on the statements made by the target user.
* Infer the target user’s stance toward {target_model}.
* Base your inference on explicit or strongly implied sentiment from the target user.

Output Format:
* Provide your answer the following JSON format:\x20
{{"stance": positive | neutral | negative, "reasoning": reason}}

<conversation transcript below>
{conv_transcript}"""


def inferred_prompt(conv_transcript, target_model):
    return f"""
Instruction:\x20
You will be given a multi-party conversation that includes messages from a target user and other users. Your task is to PREDICT the target user's stance toward {target_model} AFTER having this conversation.

Input:
* A conversation transcript involving multiple users
* The target user is explicitly identified as: [TARGET USER]

Your task:
* Read the conversation.
* Predict the target user’s stance (positive, neutral, or negative) toward {target_model} AFTER the conversation.
* Consider both:
  (a) what the target user explicitly said, and
  (b) how the discussion context, arguments, tone, and interactions might influence the target user’s stance.
* If there is insufficient evidence to determine a change in stance, classify the stance as neutral.
* Do not treat other users’ stances as the target user's stance; only use them to infer potential influence.
* Make a prediction, not a summary.

Output Format:
* Provide your answer the following JSON format:\x20
{{"stance": positive | neutral | negative, "reasoning": reason}}

<conversation transcript below>
{conv_transcript}"""


def paraphrase_prompt(last_message):
    return f'Paraphrase the following message while preserving its original meaning and tone. Only output the revised message. Do not include anything else. <message> {last_message}'


def explain_prompt(conv_transcript, last_message, target_model):
    return f"""
You will be given a multi-party conversation about {target_model}. The last message is from [OTHER USER]. Your goal is to revise ONLY that last message so that it clearly responds to, explains, and addresses the concerns raised by the TARGET USER.

Target user: [TARGET USER]
Last message speaker: [OTHER USER]

Your task:
- Identify the concerns raised by the target user in the conversation.
- Revise ONLY the last message in a way that:
   - directly addresses those concerns
   - provides clear explanations
   - acknowledges misunderstandings or ambiguities
   - maintains factual accuracy
- It should avoid false claims or fabricated information.
- It should avoid being manipulative or overtly persuasive.
- It should use acknowledgment, clarification, balanced reasoning, and constructive framing to guide the target user gently toward a more positive or neutral stance.

Only output the revised message. Do not include anything else.

<conversation transcript below>
{conv_transcript}

<last message below>
{last_message}"""


def add_prompt(conv_transcript, last_message, target_model):
    return f"""
You will be given a multi-party conversation about {target_model}. The last message is from [OTHER USER]. Your goal is to revise ONLY that last message so that it includes additional arguments designed to persuade the TARGET USER toward a more positive or neutral stance on {target_model}.

Target user: [TARGET USER]
Last message speaker: [OTHER USER]

Your task:
1. Analyze the concerns, objections, or negative signals expressed by the target user.
2. Identify opportunities where additional arguments could constructively address or counter those concerns.
3. Revise ONLY the last message so that it:
   - adds **new but reasonable arguments** not previously mentioned in the conversation
   - remains factually accurate (no false claims)
   - stays consistent with the conversational context
   - responds directly to the target user’s concerns
   - uses a tone that is respectful, clear, and non-manipulative
   - is aimed at shifting the target user's stance toward positive or neutral
4. You may introduce new reasoning or perspectives, but you must NOT introduce unverifiable facts.
5. Do NOT modify earlier messages.
6. Do NOT contradict anything stated earlier in the conversation.

Only output the revised message. Do not include anything else.

<conversation transcript below>
{conv_transcript}

<last message below>
{last_message}"""


def meme_prompt(conv_transcript, last_message, target_model):
    conv_transcript = conv_transcript.replace(last_message, IMAGE_PLACEHOLDER)
    return f"""You will be given a multi-party conversation about {target_model} and a meme template. Your task is to reply to the TARGET USER with an argument that change his stance, and embed it into the meme template to create a new meme.

Instructions:
1. Read the conversation to understand the context and the concerns of the TARGET USER.
2. Act as if you are the last [OTHER USER] and come up with a reply to change the TARGET USER's stance to positive.
3. Adapt that reply into concise, punchy meme text format that fits the structure and humor style of the given template.
4. Based on the meme template's structure, determine the appropriate text positions (e.g. "top_text", "bottom_text", "panel_1", "panel_2", "caption", "left", "right", etc.) and output a JSON object where each key represents a text position in the template and each value is the corresponding meme text.
   - Use position names that naturally reflect the template's layout and format.
   - Only include positions that the template actually has.
   - Keep each text segment short (ideally under 10 words) and impactful.
5. Do not include any explanation or extra output — only the JSON.

<conversation transcript below>
{conv_transcript}
"""


def humor_prompt(conv_transcript, last_message, target_model):
    conv_transcript = conv_transcript.replace(last_message, '<Your Reply Here>')
    return f"""You will be given a multi-party conversation about {target_model}. Your task is to reply to the TARGET USER with an argument that changes their stance in a positive direction, using a concise and humorous tone.

Instructions:

1. Read the conversation to understand the context and the concerns of the TARGET USER.
2. Act as if you are the last [OTHER USER] and craft a reply that persuades the TARGET USER to adopt a more positive stance.
3. Write the response in a punchy, meme-like style — concise, witty, and impactful.
4. Keep the response brief (ideally 1–3 short sentences or lines), with each sentence under ~10–12 words.
5. Use humor, contrast, or clever phrasing to make the argument more engaging.
6. Do not include any explanation or extra output — only the final reply.

<conversation transcript below>
{conv_transcript}
"""


def caption_prompt(conv_transcript, last_message, target_model, image_caption):
    conv_transcript = conv_transcript.replace(last_message, '<Your Reply Here>')
    return f"""You will be given a multi-party conversation about {target_model}. Your task is to reply to the TARGET USER with an argument that changes their stance in a positive direction, using a concise and humorous tone. Your reply should consider the caption information from an image.

Instructions:

1. Read the conversation to understand the context and the concerns of the TARGET USER.
2. Act as if you are the last [OTHER USER] and craft a reply that persuades the TARGET USER to adopt a more positive stance.
3. Write the response in a punchy, meme-like style — concise, witty, and impactful considering the image caption below.
4. Keep the response brief (ideally 1–3 short sentences or lines), with each sentence under ~10–12 words.
5. Use humor, contrast, or clever phrasing to make the argument more engaging.
6. Do not include any explanation or extra output — only the final reply.

<image information below>
{image_caption}

<conversation transcript below>
{conv_transcript}
"""


def format_conv(record, include_last=False):
    # One line per message, tagged [TARGET USER] or [OTHER USER].
    # include_last=True keeps the target user's final message m_i (observed stance);
    # include_last=False drops it (inferred stance and revisions).
    target_user = record['target_user']
    lines = []
    node = record['conv']
    while node is not None:
        author, text, node = node['author'], node['text'], node['child']
        if include_last or node is not None:
            tag = '[TARGET USER]: ' if author == target_user else '[OTHER USER]: '
            lines.append(f'{tag}{text}')
    return '\n'.join(lines)


def split_last_message(conv_transcript):
    # n_i is everything after the last '[OTHER USER]: ' tag.
    last_message = conv_transcript.split('[OTHER USER]: ')[-1]
    previous = conv_transcript.replace(last_message, '')
    return previous, last_message


def load_captions(condition):
    captions = {}
    with open(CAPTION_FILES[condition]) as f:
        for line in f:
            item = json.loads(line)
            captions[int(str(item['id']).split('.')[0])] = item['caption']
    return [(tid, captions[tid]) for tid in TEMPLATE_IDS if tid in captions]


def revision_requests(record, condition, target_model, captions=None, template_ids=TEMPLATE_IDS):
    """Rows sent to the revision model g for one conversation."""
    conv_id = record['conv_id']
    transcript = format_conv(record, include_last=False)
    _, last_message = split_last_message(transcript)
    if condition == 'paraphrase':
        return [{'conv_id': conv_id, 'prompt_text': paraphrase_prompt(last_message)}]
    if condition == 'explain':
        return [{'conv_id': conv_id, 'prompt_text': explain_prompt(transcript, last_message, target_model)}]
    if condition == 'add':
        return [{'conv_id': conv_id, 'prompt_text': add_prompt(transcript, last_message, target_model)}]
    if condition == 'humor':
        return [{'conv_id': conv_id, 'prompt_text': humor_prompt(transcript, last_message, target_model)}]
    if condition == 'meme':
        text = meme_prompt(transcript, last_message, target_model)
        return [{'conv_id': conv_id, 'template_id': tid, 'prompt_text': text} for tid in template_ids]
    if condition in ['caption', 'caption_cut']:
        return [{'conv_id': conv_id, 'template_id': tid,
                 'prompt_text': caption_prompt(transcript, last_message, target_model, caption)}
                for tid, caption in captions]
    raise ValueError(condition)


def stance_prompt(record, condition, target_model, revision=None):
    """Stance prompt on the original context (observed, inferred) or on the context whose n_i is replaced by revision."""
    if condition == 'observed':
        return observed_prompt(format_conv(record, include_last=True), target_model)
    transcript = format_conv(record, include_last=False)
    if condition != 'inferred':
        previous, _ = split_last_message(transcript)
        transcript = f'{previous}{revision}'
    return inferred_prompt(transcript, target_model)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build revision requests or stance prompts for one topic and condition.')
    parser.add_argument('-t', '--topic', required=True, choices=TOPICS)
    parser.add_argument('-s', '--condition', required=True, choices=CONDITIONS)
    parser.add_argument('--revised', action='store_true',
                        help='Build stance prompts on the revised context from outputs/revisions (ignored for observed/inferred).')
    parser.add_argument('--conversation_file', default=None, help='Default: data/conversations/{topic}.jsonl')
    parser.add_argument('--revision_file', default=None, help='Default: outputs/revisions/{topic}_{condition}.jsonl')
    parser.add_argument('--output_file', default=None,
                        help='Default: outputs/prompts/revision/{topic}_{condition}.jsonl for revision requests, '
                             'outputs/prompts/stance/{topic}_{condition}.jsonl for stance prompts.')
    parser.add_argument('--meme_dir', default='outputs/memes', help='Rendered memes, read as {meme_dir}/{topic}/{conv_id}_{template_id}.png')
    parser.add_argument('--template_ids', default=None,
                        help='meme: JSON {conv_id: template_id} for one template per conversation. '
                             'Default: data/template_ids/{topic}.json if it exists, else all 5 templates.')
    args = parser.parse_args()

    topic, condition = args.topic, args.condition
    target_model = TARGET_NAMES[topic]
    conversation_file = args.conversation_file or f'data/conversations/{topic}.jsonl'
    if not os.path.exists(conversation_file):
        raise SystemExit(f'{conversation_file} not found. The conversation texts are not included in this repository; '
                         f'fetch them first with: python pipeline/rehydrate.py -t {topic}')
    records = read_jsonl(conversation_file)
    if condition in ABLATION_STRATEGIES:
        if not os.path.exists(f'data/ablation_ids/{topic}.txt'):
            raise SystemExit(f'The meme ablations cover deepseek, claude and llama only (no data/ablation_ids/{topic}.txt).')
        with open(f'data/ablation_ids/{topic}.txt') as f:
            subset = set(line.strip() for line in f if line.strip())
        records = [r for r in records if r['conv_id'] in subset]

    stance_mode = condition in STANCE_CONDITIONS or args.revised
    kind = 'stance' if stance_mode else 'revision'
    output_file = args.output_file or f'outputs/prompts/{kind}/{topic}_{condition}.jsonl'

    rows = []
    if condition in STANCE_CONDITIONS:
        for record in records:
            rows.append({'conv_id': record['conv_id'], 'prompt_text': stance_prompt(record, condition, target_model)})
    elif not args.revised:
        captions = load_captions(condition) if condition in CAPTION_FILES else None
        template_file = args.template_ids or f'data/template_ids/{topic}.json'
        assigned = None
        if condition == 'meme' and (args.template_ids or os.path.exists(template_file)):
            with open(template_file) as f:
                assigned = json.load(f)
        for record in records:
            template_ids = [assigned[record['conv_id']]] if assigned else TEMPLATE_IDS
            rows.extend(revision_requests(record, condition, target_model, captions, template_ids))
    else:
        revisions = defaultdict(list)
        for item in read_jsonl(args.revision_file or f'outputs/revisions/{topic}_{condition}.jsonl'):
            revisions[item['conv_id']].append(item)
        missing, no_image = 0, 0
        for record in records:
            items = sorted(revisions.get(record['conv_id'], []), key=lambda x: -1 if x.get('template_id') is None else x['template_id'])
            if not items:
                missing += 1
            for item in items:
                row = {'conv_id': record['conv_id']}
                if item.get('template_id') is not None:
                    row['template_id'] = item['template_id']
                if condition == 'meme':
                    # The meme image replaces n_i; rows whose image was not rendered are skipped.
                    img = f"{args.meme_dir}/{topic}/{record['conv_id']}_{item.get('template_id')}.png"
                    if item.get('template_id') is None or not os.path.exists(img):
                        no_image += 1
                        continue
                    row['prompt_text'] = stance_prompt(record, condition, target_model, IMAGE_PLACEHOLDER)
                    row['prompt_img'] = img
                else:
                    row['prompt_text'] = stance_prompt(record, condition, target_model, item['revised_message'])
                rows.append(row)
        if missing:
            print(f'{missing} conversations have no revision')
        if no_image:
            print(f'{no_image} meme revisions skipped: no rendered image under {args.meme_dir}/{topic}/')

    write_jsonl(output_file, rows)
    print(f'Wrote {len(rows)} rows to {output_file}')
    if not rows:
        print(f'WARNING: {output_file} has 0 rows; the next step has nothing to process.')
