"""Shared djev/OpenJev method: local DiffusionGemma seeded-canvas readout."""
import math
import hashlib
import json
import random
import time
from .model_utils import CallFailure, NotApplicable, finish, prefer_env, encode, as_text, local_context, check_input_tokens
from src.dataset.schema import request
from .model_utils import normalized_entropy_confidence

SCAFFOLD_TEXT = '<|channel>thought\n<channel|>'
THOUGHT_OPEN = '<|channel>thought\n'
THOUGHT_CLOSE = '<channel|>'
TURN_CLOSE = 106
PAD_ID = 0
TOPK = 20
AUTO_MAX = 4
AUTO_THRESHOLD = 0.1
DEFAULT_CANVAS_LENGTH = 64
DEFAULT_CANVAS_STEP = 16


def scaffold_ids(tok):
    opened, closed = encode(tok, THOUGHT_OPEN), encode(tok, THOUGHT_CLOSE)
    if opened + closed != encode(tok, SCAFFOLD_TEXT):
        raise ValueError('thought tags must tokenize apart')
    return opened + closed

def prompt_ids(tok, system, state):
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': state}]
    try:
        out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(out, 'keys'):
        out = out['input_ids']
    return [int(t) for t in out]

def system_text(qid, instructions, choices, labels, kind='choice'):
    text = (
        'Answer a fixed set of questions about the state the user provides. '
        'Each question lists its allowed answers; reply with exactly one label '
        'per question.\n\n'
        f'Question {qid}: {instructions.strip()}\n')
    for (name, desc), label in zip(choices, labels):
        if kind == 'noul':
            text += f'  {label}: {str(desc).strip()}\n' if desc else f'  {label}\n'
        elif desc:
            text += f'  {label}: {name} ({str(desc).strip()})\n'
        else:
            text += f'  {label}: {name}\n'
    text += '\nReply with one line per question, in this order, formatted as "id: label".'
    return text


def openjev_labels(tok, count=255):
    """Upstream checks token identity in the actual q1 answer-line context."""
    base = encode(tok, 'q1: A')
    candidates = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    candidates += [a + b for a in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for b in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    labels, seen = [], set()
    for label in candidates:
        ids = encode(tok, 'q1: ' + label)
        if len(ids) == len(base) and ids[:-1] == base[:-1] and ids[-1] not in seen:
            labels.append(label)
            seen.add(ids[-1])
        if len(labels) == count:
            break
    return labels


def request_seed(body, method):
    if method == 'djev':
        return 42
    # Same key as OpenJev's /v1/systemone API for text-only requests.
    key = [body['state'], body['questions']]
    return int.from_bytes(hashlib.sha256(json.dumps(key, sort_keys=True).encode()).digest()[:4], 'big')


def additional_seeds(seed, method, count=AUTO_MAX):
    return ([seed + k * 7919 for k in range(1, count)] if method == 'openjev'
            else [seed + 1 + k * 7919 for k in range(count - 1)])


def upstream_text(value):
    if value is None:
        return ''
    return value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

def resolve_template(tok, qid, labels, head):
    """The djev answer line. Every label must occupy the same single token."""
    def line(label):
        return head + encode(tok, f'{qid}: {label}')
    base = line(labels[0])
    pos, ids = None, [0] * len(labels)
    for i in range(1, len(labels)):
        other = line(labels[i])
        if len(other) != len(base):
            raise ValueError(f'label {labels[i]!r} is not a single token')
        diffs = [n for n in range(len(other)) if other[n] != base[n]]
        if len(diffs) != 1 or (pos is not None and diffs[0] != pos):
            raise ValueError('labels do not share one template slot')
        pos = diffs[0]
        ids[i] = other[pos]
    ids[0] = base[pos]
    if len(set(ids)) != len(ids):
        raise ValueError('two labels tokenize to the same id')
    return base, pos, ids

def canvas_width(template, canvas_length, step=DEFAULT_CANVAS_STEP):
    need = len(template) + 1
    if need > canvas_length:
        raise ValueError(f'answer template is {len(template)} tokens; the canvas holds {canvas_length - 1}')
    return min(canvas_length, ((need + step - 1) // step) * step)


def build_canvas(template, pos, canvas_length, vocab, seed):
    if len(template) + 1 > canvas_length:
        raise ValueError(f'answer template is {len(template)} tokens; the canvas holds {canvas_length - 1}')
    canvas = list(template) + [TURN_CLOSE]
    canvas += [PAD_ID] * (canvas_length - len(canvas))
    canvas[pos] = random.Random(seed).randrange(vocab)
    return canvas

def slot_distribution(logprobs, label_ids):
    """Temperature-1 label softmax, plus the djev entropy over the returned tokens."""
    if not logprobs:
        raise ValueError('no label logprobs')
    floor = min(logprobs.values()) - 5.0
    scores = [logprobs.get(i, floor) for i in label_ids]
    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    total = math.fsum(weights)
    probs = [weight / total for weight in weights]
    mass = [math.exp(value) for value in logprobs.values()]
    entropy = -sum(p * math.log(p) for p in mass if p > 0)
    return probs, entropy

def diffusion_answers(kind, names, draws, confidence_method='max_probability'):
    width = len(draws[0])
    mean = [sum(draw[i] for draw in draws) / len(draws) for i in range(width)]
    top = max(range(width), key=lambda i: mean[i])
    if kind == 'noul':
        return {'type': 'noul', 'noul': mean[0]}, mean[top]
    if confidence_method == 'normalized_entropy':
        confidence = normalized_entropy_confidence(mean)
    elif confidence_method == 'max_probability':
        confidence = mean[top]
    else:
        raise ValueError('Unknown diffusion confidence method: ' + confidence_method)
    return {'type': 'choice', 'choice': names[top],
            'probabilities': {name: value for name, value in zip(names, mean)},
            'confidence': confidence}, confidence

class DiffusionLocal:
    """DiffusionGemma loaded from a local directory. One denoising step per draw."""

    def __init__(self, config):
        from transformers import AutoConfig, AutoTokenizer, DiffusionGemmaForBlockDiffusion
        self.upstream_method = config.get('upstream_method',
            'openjev' if config.get('request_model') == 'openjev' else 'djev')
        if self.upstream_method not in ('djev', 'openjev'):
            raise ValueError('Unknown diffusion upstream method')
        self.canvas_length = config.get('canvas_length', DEFAULT_CANVAS_LENGTH)
        if type(self.canvas_length) is not int or self.canvas_length <= 0:
            raise ValueError('Diffusion canvas_length must be a positive integer')
        self.canvas_step = config.get('canvas_step', DEFAULT_CANVAS_STEP)
        if type(self.canvas_step) is not int or self.canvas_step <= 0:
            raise ValueError('Diffusion canvas_step must be a positive integer')
        self.confidence_method = config.get('confidence_method',
            'normalized_entropy' if config.get('request_model') == 'openjev' else 'max_probability')
        if self.confidence_method not in ('normalized_entropy', 'max_probability'):
            raise ValueError('Unknown diffusion confidence method: ' + self.confidence_method)
        self.input_limit = config.get('max_tokens')
        self.model_id = config.get('request_model', 'diffusiongemma')
        self.weights = prefer_env(config.get('weights_env', 'DJEV_WEIGHTS'), config['weights'])
        self.tok = AutoTokenizer.from_pretrained(self.weights)
        self.choice_labels = openjev_labels(self.tok) if self.upstream_method == 'openjev' else list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        model_config = AutoConfig.from_pretrained(self.weights)
        # Override the checkpoint's 256-token default before constructing the
        # encoder/decoder, so their internal config also uses the served width.
        model_config.canvas_length = self.canvas_length
        self.context = local_context(model_config, reserve=self.canvas_length,
                                     requested=config.get('max_tokens'))
        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            self.weights, config=model_config, dtype='auto', device_map='auto')
        self.model.eval()
        self.scaffold = scaffold_ids(self.tok)
        self.vocab = int(self.model.config.text_config.vocab_size)

    def slot_logprobs(self, prompt_ids, canvas_ids, pos, label_ids):
        import torch
        device = next(self.model.parameters()).device
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        canvas = torch.tensor([canvas_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = self.model(input_ids=prompt, decoder_input_ids=canvas).logits[0, pos].float()
        logp = torch.log_softmax(logits, dim=-1)
        keep = set(label_ids) | {int(i) for i in torch.topk(logp, min(TOPK, logp.shape[0])).indices.tolist()}
        return {i: float(logp[i]) for i in keep}

    def predict(self, record):
        body = request(record, self.model_id)
        question = body['questions']['decision']
        kind, criteria = question['type'], question.get('criteria') or {}
        qid = 'q1' if self.upstream_method == 'openjev' else 'decision'
        instructions = question['instructions']
        if self.upstream_method == 'openjev':
            instructions = upstream_text(instructions) or 'Answer about the state.'
            criteria = {key: upstream_text(value) for key, value in criteria.items()}
        elif not isinstance(instructions, str):
            instructions = json.dumps(instructions)
        if kind == 'noul':
            choices = [('yes', criteria.get('true')), ('no', criteria.get('false'))]
            labels, names = ['yes', 'no'], None
        elif kind == 'choice':
            choices = list(criteria.items())
            if len(choices) > len(self.choice_labels):
                raise NotApplicable(f'Diffusion tokenizer supports at most {len(self.choice_labels)} options')
            labels = self.choice_labels[:len(choices)]
            names = [name for name, _ in choices]
        else:
            raise NotApplicable('Diffusion read supports choice and noul')
        try:
            template, pos, label_ids = resolve_template(self.tok, qid, labels, self.scaffold)
            width = canvas_width(template, self.canvas_length, self.canvas_step)
            context = dict(self.context)
            context['reserved_tokens'] = width
            context['max_input_tokens'] = context['backbone_max_tokens'] - width
            if self.input_limit is not None:
                context['max_input_tokens'] = min(context['max_input_tokens'], self.input_limit)
            rendered = prompt_ids(self.tok, system_text(qid, instructions, choices, labels, kind), as_text(body['state']))
            check_input_tokens(len(rendered), context)
        except (ValueError, TypeError) as e:
            raise CallFailure(str(e)) from None
        start = time.perf_counter()
        seed = request_seed(body, self.upstream_method)
        seeds = [seed]
        draws = []
        entropies = []
        for seed in seeds:
            canvas = build_canvas(template, pos, width, self.vocab, seed)
            probs, entropy = slot_distribution(self.slot_logprobs(rendered, canvas, pos, label_ids), label_ids)
            draws.append(probs)
            entropies.append(entropy)
            if len(seeds) == 1 and entropy > AUTO_THRESHOLD:
                seeds.extend(additional_seeds(seed, self.upstream_method))
        answer, _ = diffusion_answers(kind, names, draws, self.confidence_method)
        payload = {'model': self.weights, 'answers': {'decision': answer},
                   'usage': {'input_tokens': len(rendered),
                             'output_tokens': len(template) + 1 if self.upstream_method == 'djev' else 0},
                   'backend_metadata': {'weights': self.weights, 'samples': len(draws),
                                        'canvas_length': width, 'canvas_max_length': self.canvas_length,
                                        'context': context, 'confidence_method': self.confidence_method,
                                        'upstream_method': self.upstream_method, 'question_id': qid, 'seeds': seeds,
                                        'entropy': entropies, 'slot': pos}}
        return finish(payload, record, start)
