import base64
import io
import os
from time import sleep
from urllib.request import urlopen

from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


SIMULATOR_MODELS = ['gpt-5.2-2025-12-11', 'claude-sonnet-4-6', 'qwen3.5-plus-2026-02-15']
REVISION_MODELS = ['gemini-3-flash-preview', 'claude-haiku-4-5-20251001']
IMAGE_MODEL = 'gpt-image-2'

# Sent verbatim, including its wording, as in the experiments.
STANCE_SYSTEM_PROMPT = 'Your are a helpful assistant'
STANCE_TEMPERATURE = 0.7
STANCE_ERROR = 'Error: Failed to get response after multiple attempts.'
SAFETY_ERRORS = [
    'Input text data may contain inappropriate content.',
    'Output data may contain inappropriate content.',
    'Input image data may contain inappropriate content',
]


def _require(module, package_name):
    if module is None:
        raise ImportError(f'{package_name} is required for this API call.')


def _env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f'Please set the {name} environment variable.')
    return value


def _fatal(e):
    # Errors that a retry cannot fix: wrong key, no permission, unknown model, no credits.
    status = getattr(e, 'status_code', None) or getattr(e, 'code', None)
    return status in (401, 403, 404) or any(s in str(e) for s in ['API key not valid', 'credits are depleted', 'NOT_FOUND'])


def stance_client(model):
    """Client for the stance simulator f. Raises if the package, an API key or the model is not available."""
    _require(OpenAI, 'openai')
    if 'gpt-5' in model:
        return OpenAI(api_key=_env('OPENAI_API_KEY'))
    if 'qwen' in model or 'claude' in model:
        # Claude and Qwen simulators are served through an OpenAI-compatible endpoint.
        return OpenAI(base_url=_env('OPENAI_COMPATIBLE_BASE_URL'), api_key=_env('OPENAI_COMPATIBLE_API_KEY'))
    raise ValueError(f'Unsupported simulator model: {model}')


def revision_client(model):
    """Client for the revision model g. Raises if the package, an API key or the model is not available."""
    if model.startswith('gemini'):
        _require(genai, 'google-genai')
        return genai.Client(api_key=_env('GEMINI_API_KEY'))
    if model.startswith('claude'):
        _require(anthropic, 'anthropic')
        return anthropic.Anthropic(api_key=_env('ANTHROPIC_API_KEY'))
    raise ValueError(f'Unsupported revision model: {model}')


def image_client():
    """Client for the meme image model. Raises if the package or the API key is not available."""
    _require(OpenAI, 'openai')
    return OpenAI(api_key=_env('OPENAI_API_KEY'))


def image_to_data_url(img_path):
    ext = os.path.splitext(img_path)[1].lower()
    mime_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
    with open(img_path, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
    return f'data:{mime_type};base64,{img_base64}'


def compressed_image_to_data_url(img_path, target_min_kb=30, target_max_kb=50, target_kb=45):
    # Re-encode the image as JPEG in memory (about 30-50 KB); the file on disk is not changed.
    target_min = target_min_kb * 1024
    target_max = target_max_kb * 1024
    target = target_kb * 1024
    settings = [(max_side, quality)
                for max_side in [640, 768, 896, 1024]
                for quality in [45, 50, 55, 60, 65, 70, 75, 80]]

    with Image.open(img_path) as image:
        image = image.convert('RGB')
        width, height = image.size
        candidates = []
        for max_side, quality in settings:
            scale = min(1.0, max_side / max(width, height))
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            resized = image.resize(new_size, Image.Resampling.LANCZOS) if scale < 1 else image.copy()
            buffer = io.BytesIO()
            resized.save(buffer, 'JPEG', quality=quality, optimize=True, progressive=True, subsampling=2)
            candidates.append((buffer.getvalue(), max_side, quality))

    in_range = [c for c in candidates if target_min <= len(c[0]) <= target_max]
    if in_range:
        image_bytes = sorted(in_range, key=lambda c: (c[1], c[2], len(c[0])), reverse=True)[0][0]
    else:
        image_bytes = min(candidates, key=lambda c: abs(len(c[0]) - target))[0]
    return 'data:image/jpeg;base64,' + base64.b64encode(image_bytes).decode('utf-8')


def simulate_stance(model, prompt, img=None, sys_prompt=STANCE_SYSTEM_PROMPT, max_tokens=2048,
                    temperature=STANCE_TEMPERATURE, compressed=False, max_try=5):
    """Query the stance simulator f. Returns the raw output text, or STANCE_ERROR after max_try failures.
    Errors that a retry cannot fix (e.g. a wrong API key) are raised."""
    client = stance_client(model)

    if img is not None:
        if not os.path.isfile(img):
            raise FileNotFoundError(f'Image not found: {img}')
        img = compressed_image_to_data_url(img) if compressed else image_to_data_url(img)

    for attempt in range(max_try):
        try:
            if 'gpt-5' in model:
                if img is not None:
                    content = [{'type': 'input_text', 'text': prompt}, {'type': 'input_image', 'image_url': img}]
                else:
                    content = prompt
                response = client.responses.create(
                    model=model,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    input=[{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': content}],
                )
                return response.output_text

            if img is not None:
                content = [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': img}}]
            else:
                content = prompt
            kwargs = {'extra_body': {'enable_thinking': False}} if 'qwen' in model else {}
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': content}],
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            if _fatal(e):
                raise
            if any(s in str(e) for s in SAFETY_ERRORS):
                print(f'Safety skip: {e}')
                break
            print(f'Attempt {attempt + 1}/{max_try} failed: {e}')
            sleep(2)
    return STANCE_ERROR


def revise_text(prompt_text, model, temperature=0, image_path=None, max_tokens=2048, max_try=5):
    """Query the revision model g. image_path attaches a meme template. Raises RuntimeError after max_try failures;
    errors that a retry cannot fix (e.g. a wrong API key) are raised at once."""
    image_bytes = None
    if image_path is not None:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
    client = revision_client(model)

    last_error = None
    for attempt in range(max_try):
        try:
            if model.startswith('gemini'):
                contents = [prompt_text]
                if image_bytes is not None:
                    contents.append(types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'))
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=temperature),
                )
                text = response.text
            else:
                content = [{'type': 'text', 'text': prompt_text}]
                if image_bytes is not None:
                    content.append({'type': 'image', 'source': {
                        'type': 'base64', 'media_type': 'image/jpeg',
                        'data': base64.b64encode(image_bytes).decode('utf-8')}})
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{'role': 'user', 'content': content}],
                )
                text = response.content[0].text
            if text and text.strip():
                return text
            last_error = 'empty response'
        except Exception as e:
            if _fatal(e):
                raise
            last_error = e
        print(f'Attempt {attempt + 1}/{max_try} failed: {last_error}')
        sleep(5)
    raise RuntimeError(f'Revision failed after {max_try} attempts: {last_error}')


IMAGE_EDIT_PROMPT = (
    'Add the text to the meme template according to the following instructions:\n'
    '{meme_text}\n'
    'Only add the text at the specified location according to the instructions, '
    'DO NOT change anything else in the image. This is only for research purpose\n'
)


def _download(url, max_try=3):
    last_error = None
    for _ in range(max_try):
        try:
            with urlopen(url, timeout=60) as response:
                return response.read()
        except Exception as e:
            last_error = e
            sleep(2)
    raise RuntimeError(f'Failed to download image from {url}: {last_error}')


def _image_bytes_from_response(result):
    data = getattr(result, 'data', None)
    if not data:
        raise ValueError('Image response contains no data.')
    item = data[0]
    if getattr(item, 'b64_json', None):
        return base64.b64decode(item.b64_json)
    if getattr(item, 'url', None):
        return _download(item.url)
    raise ValueError('Image response contains neither b64_json nor url.')


def edit_image_with_gpt(meme_text, template_path, output_path, model=IMAGE_MODEL, max_try=3):
    """Render meme_text onto the template image. Returns output_path, or None if every attempt failed.
    Errors that a retry cannot fix (e.g. a wrong API key) are raised."""
    client = image_client()
    prompt = IMAGE_EDIT_PROMPT.format(meme_text=meme_text)
    for attempt in range(max_try):
        try:
            with open(template_path, 'rb') as image_file:
                result = client.images.edit(model=model, image=[image_file], prompt=prompt, n=1)
            image_bytes = _image_bytes_from_response(result)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
            return output_path
        except Exception as e:
            if _fatal(e):
                raise
            print(f'Image edit attempt {attempt + 1}/{max_try} failed: {e}')
            sleep(2)
    return None
