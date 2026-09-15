import asyncio
import copy
import gzip
import json

import pytest

from data_synthesis.config import load_config
from data_synthesis.contract import text_pair
from data_synthesis import reuse_farm
from data_synthesis.review_contract import VERSION as REVIEW_VERSION
from data_synthesis.review_farm import ReviewDB, export_review, read_batch, review_items


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()


def config():
    c = copy.deepcopy(load_config())
    c.update(review_mode='reuse_first_targeted', review_contract=reuse_farm.VERSION,
             previous_deepseek_roots=[], routing_workers=1)
    c['sii'].update(concurrency_start=1, concurrency_max=2, max_attempts=1)
    return c


def item(key='one', text='A red circle on a white background.'):
    candidate = {'image_identity': 'blip3o_long:original', 'text': text, 'kind': 'curated_caption',
                 'author': 'original-caption-model', 'provenance': {'dataset': 'original', 'field': 'one.txt'}}
    return {'key': key, 'original_key': key, 'row': {'source': 'blip3o_long', 'source_id': 'original',
            'split': 'train', 'caption_candidates': [candidate]}, 'group': 'long_cc12m',
            'cohort': 'blip3o_long', 'batch_id': 'batch', 'source_run': '/source',
            'view': {'source_path': '/image.jpg', 'view_id': '/image.jpg', 'hashes_computed': False,
                     'original_size': [512, 512], 'crop': [0, 0, 512, 512]}}


def raw(i, contract, model=reuse_farm.MODEL):
    result = {'review': {'decision': 'rewrite', 'candidate_index': None, 'issues': ['missing caption']},
              'pair': text_pair(i['key'], 'A red circle on a white background.')}
    return {'status': 'completed', 'backend': 'sii', 'requested_model': model, 'returned_model': model,
            'contract_hash': contract, 'image_id': i['key'], 'view_id': '/image.jpg',
            'decoded_size': [512, 512], 'image_attached': True, 'output_text': json.dumps(result)}


def test_original_long_text_and_author_are_preserved_without_precision_labels():
    i = item(text='  Three objects in a scene.\r\n')
    pair, evidence, issues = reuse_farm.local_route(i, config(), Tokenizer())
    assert pair['i2t'] == pair['t2i'] == 'Three objects in a scene.'
    assert pair['observations']['counts'] == []
    assert evidence['generator_models']['i2t'] == 'original-caption-model'
    assert evidence['route'] == 'normalize' and not issues


@pytest.mark.parametrize('reason', ['missing', 'wrong_identity', 'crop', 'too_long', 'known_error'])
def test_only_ineligible_or_explicitly_flagged_text_needs_sii(reason):
    i = item()
    if reason == 'missing':
        i['row']['caption_candidates'] = []
    elif reason == 'wrong_identity':
        i['row']['caption_candidates'][0]['image_identity'] = 'other:image'
    elif reason == 'crop':
        i['view']['crop'] = [1, 0, 511, 512]
    elif reason == 'too_long':
        i['row']['caption_candidates'][0]['text'] = 'word ' * 961
    else:
        i['row']['known_caption_issues'] = ['explicit confirmed error']
    pair, evidence, issues = reuse_farm.local_route(i, config(), Tokenizer())
    assert pair is None and evidence is None and issues


def test_annotations_require_exact_view_and_exhaustive_visible_referent():
    i = item()
    i['row']['caption_candidates'] = []
    fact = {'type': 'count', 'entity': 'circles', 'count': 2, 'verified': True,
            'fully_visible': True, 'exhaustive_for_referent': True,
            'view_id': '/image.jpg', 'provenance': {'dataset': 'annotated'}}
    i['row']['verified_facts'] = [fact]
    pair, evidence, _ = reuse_farm.local_route(i, config(), Tokenizer())
    assert evidence['route'] == 'annotation' and pair['observations']['counts'][0]['count'] == 2
    fact['view_id'] = '/other.jpg'
    assert reuse_farm.local_route(i, config(), Tokenizer())[0] is None


def test_routing_separates_outputs_and_resumes_without_reopening_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(reuse_farm, '_tokenizer', Tokenizer())
    monkeypatch.setattr(reuse_farm, '_overrides', {})
    a, b = item('one'), item('two', text='')
    monkeypatch.setattr(reuse_farm, 'read_batch', lambda *args, **kwargs: [a, b])
    groups = [{'records': 2}]
    result = reuse_farm.route_shard(0, groups, tmp_path, config())
    assert result['counts'] == {'reuse': 1, 'needs_sii': 1}
    with gzip.open(result['reused']['path'], 'rt') as f:
        saved = json.loads(next(f))
    assert saved['generator_models']['i2t'] == 'original-caption-model'
    with gzip.open(result['repairs']['path'], 'rt') as f:
        queued = json.loads(next(f))
    assert queued['key'] == 'two' and queued['routing_issues']
    monkeypatch.setattr(reuse_farm, 'read_batch', lambda *a, **kw: pytest.fail('must resume sealed output'))
    assert reuse_farm.route_shard(0, groups, tmp_path, config()) == result


def test_targeted_api_contract_is_validated_and_exported(tmp_path):
    c, i = config(), item(text='')
    class Client:
        async def generate(self, value, number):
            return raw(value, reuse_farm.VERSION)
    state = ReviewDB(tmp_path, c)
    try:
        state.admit([i])
        result = asyncio.run(review_items(state, Client(), c, Tokenizer(), asyncio.Event()))
        assert result['counts'] == {'ready': 1}
        manifest = export_review(state, Tokenizer())
        with gzip.open(manifest['output'], 'rt') as f:
            value = json.loads(next(f))
        assert value['contract'] == reuse_farm.VERSION and value['model'] == reuse_farm.MODEL
    finally:
        state.close()


def test_preserved_deepseek_response_can_be_reused_but_qwen_cannot(tmp_path):
    i = item()
    i['review_shard'] = '00000'
    c = config()
    c.pop('review_contract')
    state = ReviewDB(tmp_path/'shards/00000', c)
    try:
        state.admit([i])
        for model, accepted in [('qwen3.8-max', False), (reuse_farm.MODEL, True)]:
            response = raw(i, REVIEW_VERSION, model)
            state.db.execute("UPDATE items SET status='ready',attempts=1,result=?", (response['output_text'],))
            state.db.execute('INSERT OR REPLACE INTO attempts VALUES (?,?,?)',
                             (i['key'], 1, gzip.compress(json.dumps(response).encode())))
            state.db.commit()
            result, evidence = reuse_farm.cached_review(i, [tmp_path], Tokenizer())
            assert bool(result) == accepted
            if accepted:
                assert evidence['model'] == reuse_farm.MODEL and evidence['contract'] == REVIEW_VERSION
    finally:
        state.close()


def test_source_reader_keeps_annotations_only_when_requested(tmp_path):
    import sqlite3
    i = item()
    i['row']['annotations'] = [{'type': 'count', 'count': 3}]
    with sqlite3.connect(tmp_path/'state.sqlite3') as db:
        db.execute('CREATE TABLE tasks(key,row,view,status)')
        db.execute('INSERT INTO tasks VALUES (?,?,?,?)',
                   ('one', json.dumps(i['row']), json.dumps(i['view']), 'prepared'))
    batch = {'source_run': str(tmp_path), 'records': 1, 'batch_id': 'batch', 'cohort': 'blip3o_long'}
    assert 'annotations' not in read_batch(batch)[0]['row']
    assert read_batch(batch, keep_source_fields=True)[0]['row']['annotations'][0]['count'] == 3


def test_acceptance_binds_reuse_policy_and_does_not_fake_semantic_pass(tmp_path):
    c = config()
    selection, receipt = tmp_path/'selection.json', tmp_path/'acceptance.json'
    value = {'status': 'user_accepted_for_generation', 'authorized_by': 'user', 'config': c,
             'contract': reuse_farm.VERSION, 'selection': str(selection.resolve()),
             'semantic_quality_passed': False, 'codex_enabled_this_stage': False}
    receipt.write_text(json.dumps(value))
    assert reuse_farm.validate_run(c, selection, receipt) == value
    value['config'] = {**c, 'review_mode': 'all_images'}
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        reuse_farm.validate_run(c, selection, receipt)


@pytest.mark.parametrize('timeout_amendment', [False, True])
def test_parallel_routing_groups_repairs_and_resume_makes_no_new_calls(tmp_path, monkeypatch, timeout_amendment):
    from concurrent.futures import ThreadPoolExecutor
    import sqlite3
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    monkeypatch.setattr(reuse_farm, 'initialize_worker',
                        lambda *a: setattr(reuse_farm, '_tokenizer', Tokenizer()))
    monkeypatch.setattr(reuse_farm, 'ProcessPoolExecutor', lambda **kw: ThreadPoolExecutor(
        max_workers=kw['max_workers'], initializer=kw['initializer'], initargs=kw['initargs']))
    monkeypatch.setattr(reuse_farm, 'load_sii_settings', lambda: None)
    calls = []
    class Client:
        def __init__(self, *a, **kw):
            pass
        async def generate(self, i, number):
            calls.append(i['key'])
            if timeout_amendment and i['original_key'] == '0':
                return {'backend': 'sii', 'status': 'failed', 'error_type': 'transport', 'error': 'TimeoutError:'}
            return raw(i, reuse_farm.VERSION)
        async def close(self):
            pass
    monkeypatch.setattr(reuse_farm, 'SIIClient', Client)
    codex_calls = []
    class Codex:
        def __init__(self, *a, **kw):
            pass
        async def generate(self, i, number):
            codex_calls.append(i['key'])
            response = raw(i, reuse_farm.VERSION, 'gpt-5.6-sol')
            response.update(backend='codex_fallback', reasoning_effort='low', exit_code=0,
                command=['codex', 'exec', '--model', 'gpt-5.6-sol', '-c', 'model_reasoning_effort="low"',
                         '--image', '/image.jpg', '--ephemeral', '--ignore-user-config', '--output-schema', '/schema.json', '--json'],
                events='\n'.join(json.dumps(e) for e in [
                    {'type': 'thread.started', 'thread_id': 'fixture'},
                    {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': response['output_text']}},
                    {'type': 'turn.completed'}]))
            return response
        async def close(self):
            pass
    monkeypatch.setattr('data_synthesis.timeout_fallback.CodexFallback', Codex)
    batches = []
    for n in range(128):
        path = tmp_path/'sources'/str(n)
        path.mkdir(parents=True)
        value = item(str(n), text='' if n in (0, 1, 64, 65) else 'A red circle.')
        with sqlite3.connect(path/'state.sqlite3') as db:
            db.execute('CREATE TABLE tasks(key,row,view,status)')
            db.execute('INSERT INTO tasks VALUES (?,?,?,?)',
                       (str(n), json.dumps(value['row']), json.dumps(value['view']), 'prepared'))
        batches.append({'source_run': str(path), 'records': 1, 'batch_id': str(n), 'cohort': 'blip3o_long'})
    index = tmp_path/'index.jsonl'
    index.write_text(''.join(json.dumps(b)+'\n' for b in batches))
    selection, receipt = tmp_path/'selection.json', tmp_path/'acceptance.json'
    selection.write_text(json.dumps({'downloads_enabled': False, 'compute_hashes': False,
        'images': 128, 'batch_index': str(index), 'batch_index_bytes': index.stat().st_size,
        'image_count_basis': 'test'}))
    c = config()
    c.update(repair_overrides=None, repair_batch_images=4)
    if timeout_amendment:
        c['sii']['max_attempts'] = 2
    receipt.write_text(json.dumps({'status': 'user_accepted_for_generation', 'authorized_by': 'user',
        'contract': reuse_farm.VERSION, 'config': c, 'selection': str(selection.resolve()),
        'semantic_quality_passed': False, 'codex_enabled_this_stage': False}))
    root = tmp_path/'run'
    if timeout_amendment:
        from data_synthesis.timeout_fallback import POLICY_VERSION
        root.mkdir()
        (root/'timeout_fallback_policy.json').write_text(json.dumps({
            'version': POLICY_VERSION, 'authorized_by': 'user', 'user_instructions': ['GPT-5.6-sol low for repeated timeouts'],
            'parent_root': str(root.resolve()), 'parent_contract': reuse_farm.VERSION, 'model': 'gpt-5.6-sol',
            'reasoning_effort': 'low', 'consecutive_timeouts': 2, 'max_attempts': 1, 'concurrency': 2,
            'timeout_seconds': 300, 'compute_hashes': False}))
    for _ in range(2):
        result = asyncio.run(reuse_farm.run(root, c, selection, receipt))
        assert result['state'] == 'completed' and result['completed_images'] == 128
        assert result['routes'] == {'reuse': 124, 'needs_sii': 4}
        assert len(calls) == (5 if timeout_amendment else 4)
        assert len(codex_calls) == int(timeout_amendment)
        if timeout_amendment:
            assert result['timeout_fallback']['counts'] == {'ready': 1}
            assert result['repair_finalized_counts'] == {'deferred_codex': 1, 'ready': 3}
    assert json.loads((root/'repair_groups/00000.json').read_text())['source_shards'] == [0, 1]
