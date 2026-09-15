import asyncio
import copy
import gzip
import json

import pytest

from data_synthesis.config import load_config
from data_synthesis.contract import text_pair
from data_synthesis.review_contract import parse_codex_review
from data_synthesis.review_farm import ReviewDB, review_items, export_review
from data_synthesis.reuse_farm import VERSION, prompt_for_targeted
from data_synthesis.timeout_fallback import POLICY_VERSION, load_timeout_policy, repeated_timeouts, run_timeout_fallback


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()


def config():
    c = copy.deepcopy(load_config())
    c['sii'].update(concurrency_start=2, concurrency_max=2)
    return c


def item():
    return {'key': 'one', 'original_key': 'one', 'row': {'source': 'test', 'caption_candidates': []},
            'view': {'source_path': '/image.jpg', 'view_id': '/image.jpg', 'hashes_computed': False},
            'group': 'test', 'source_run': '/source'}


def timeout():
    return dict(backend='sii', status='failed', error_type='transport', error='TimeoutError:')


def policy(root):
    return dict(version=POLICY_VERSION, authorized_by='user', user_instructions=['反复超时的直接用codex cli进行',
        '调用GPT-5.6-sol low'], parent_root=str(root.resolve()), parent_contract=VERSION,
        model='gpt-5.6-sol', reasoning_effort='low', consecutive_timeouts=2, concurrency=2,
        max_attempts=1, timeout_seconds=300, executable='codex', compute_hashes=False)


def seed(state, responses, status='retry'):
    state.admit([item()])
    for n, value in enumerate(responses, 1):
        state.db.execute('INSERT INTO attempts VALUES (?,?,?)', ('one', n, gzip.compress(json.dumps(value).encode())))
    state.db.execute('UPDATE items SET status=?,attempts=?', (status, len(responses)))
    state.db.commit()


def codex_raw(i):
    output = json.dumps({'review': {'decision': 'rewrite', 'candidate_index': None, 'issues': ['missing caption']},
                         'pair': text_pair(i['key'], 'A red circle on a white background.')})
    return dict(status='completed', backend='codex_fallback', requested_model='gpt-5.6-sol',
        reasoning_effort='low', exit_code=0, contract_hash=VERSION, image_id=i['key'], view_id='/image.jpg',
        image_attached=True, decoded_size=[512, 512], output_text=output,
        command=['codex', 'exec', '--model', 'gpt-5.6-sol', '-c', 'model_reasoning_effort="low"',
                 '--image', '/image.jpg', '--ephemeral', '--ignore-user-config', '--output-schema', '/schema.json', '--json'],
        events='\n'.join(json.dumps(e) for e in [
            {'type': 'thread.started', 'thread_id': 'test'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': output}},
            {'type': 'turn.completed', 'usage': {'output_tokens': 30}}]))


@pytest.mark.parametrize('responses,expected', [
    ([timeout(), timeout()], True),
    ([timeout()], False),
    ([timeout(), {'status': 'failed', 'error_type': 'content'}, timeout()], False),
    ([{'status': 'failed', 'error_type': 'transport', 'error': 'HTTPStatusError: 429'}] * 2, False),
])
def test_only_consecutive_timeouts_qualify(tmp_path, responses, expected):
    state = ReviewDB(tmp_path, config())
    try:
        seed(state, responses)
        assert repeated_timeouts(state.db, 'one', len(responses)) is expected
    finally:
        state.close()


def test_saved_timeouts_leave_sii_queue_without_an_extra_api_call(tmp_path):
    class NeverCall:
        async def generate(self, *args):
            raise AssertionError('repeated timeouts must leave the primary API')
    c = config()
    state = ReviewDB(tmp_path / 'repairs' / '00000', c)
    try:
        seed(state, [timeout(), timeout()])
        result = asyncio.run(review_items(state, NeverCall(), c, Tokenizer(), asyncio.Event(), timeout_policy=policy(tmp_path)))
        assert result['counts'] == {'deferred_codex': 1}
        assert state.db.execute('SELECT count(*) FROM attempts').fetchone()[0] == 2
        exported = export_review(state, Tokenizer())
        assert exported['state'] == 'awaiting_codex'
        assert exported['outcomes'] == {'deferred_codex': 1}
    finally:
        state.close()


def test_timeout_queue_preserves_provenance_exports_and_resumes_without_calls(tmp_path):
    class Client:
        calls = 0
        async def generate(self, i, number):
            self.calls += 1
            return codex_raw(i)
        async def close(self):
            pass
    c = config()
    parent = ReviewDB(tmp_path / 'repairs' / '00000', c)
    seed(parent, [timeout(), timeout()], 'deferred_codex')
    parent.close()
    p = policy(tmp_path)
    (tmp_path / 'timeout_fallback_policy.json').write_text(json.dumps(p))
    assert load_timeout_policy(tmp_path, VERSION) == p
    client = Client()
    async def run():
        done = asyncio.Event(); done.set()
        return await run_timeout_fallback(tmp_path, p, c, Tokenizer(), asyncio.Event(), done, {}, prompt_for_targeted, client=client)
    for _ in range(2):
        assert asyncio.run(run())['counts'] == {'ready': 1}
        assert client.calls == 1
    with gzip.open(tmp_path / 'timeout_fallback' / 'reviewed.jsonl.gz', 'rt') as f:
        value = json.loads(f.readline())
    assert value['model'] == 'gpt-5.6-sol' and value['reasoning_effort'] == 'low'
    assert value['timeout_handoff']['sii_attempts'] == 2
    assert value['backend'] == 'codex_fallback'


@pytest.mark.parametrize('field,value', [('requested_model', 'other'), ('reasoning_effort', 'high'),
    ('image_attached', False), ('view_id', '/other.jpg'), ('events', ''), ('command', ['codex'])])
def test_codex_requires_fixed_model_effort_real_attachment_and_events(field, value):
    raw = codex_raw(item()); raw[field] = value
    with pytest.raises(ValueError):
        parse_codex_review(raw, item(), Tokenizer(), expected_contract=VERSION)


def test_policy_amendment_cannot_enable_an_unapproved_model(tmp_path):
    p = policy(tmp_path); p['model'] = 'other'
    (tmp_path / 'timeout_fallback_policy.json').write_text(json.dumps(p))
    with pytest.raises(ValueError):
        load_timeout_policy(tmp_path, VERSION)
