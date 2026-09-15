import asyncio
import copy
import gzip
import json
from types import SimpleNamespace

import pytest

from data_synthesis.config import load_config
from data_synthesis.contract import text_pair
from data_synthesis.review_contract import VERSION, parse_review
from data_synthesis.review_farm import ReviewDB, export_review, review_items, run_farm, validate_qualification
from data_synthesis.review_farm import update_transport_health


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()


def config():
    c = copy.deepcopy(load_config())
    c.pop('review_contract', None)  # Historical full-review prompt contract remains independently pinned.
    c['review_mode'] = 'all_images'
    c['compute_hashes'] = False
    c['codex_fallback']['enabled'] = False
    c['sii'].update(vision_models=['deepseek-v4.1-flash'], concurrency_start=2,
                    concurrency_max=2, max_attempts=2)
    return c


def item(key='one'):
    return {'key': key, 'original_key': key, 'row': {'source': 'test',
            'caption_candidates': [{'text': 'A red circle on a white background.'}]},
            'view': {'source_path': '/frozen/one.jpg', 'view_id': '/frozen/one.jpg',
                     'hashes_computed': False}, 'group': 'test', 'source_run': '/source'}


def raw(i, decision='keep'):
    text = i['row']['caption_candidates'][0]['text']
    result = {'review': {'decision': decision, 'candidate_index': 0,
                         'issues': []}, 'pair': text_pair(i['key'], text)}
    return {'backend': 'sii', 'status': 'completed', 'requested_model': 'deepseek-v4.1-flash',
            'returned_model': 'deepseek-v4.1-flash', 'contract_hash': VERSION,
            'image_id': i['key'], 'image_attached': True, 'decoded_size': [512, 512],
            'view_id': '/frozen/one.jpg', 'output_text': json.dumps(result)}


def test_review_checks_model_attachment_identity_and_exact_keep():
    i = item()
    assert parse_review(raw(i), i, Tokenizer())['review']['decision'] == 'keep'
    for field, value in [('image_attached', False), ('returned_model', 'other'),
                         ('image_id', 'wrong'), ('view_id', '/other.jpg')]:
        r = raw(i)
        r[field] = value
        with pytest.raises(ValueError):
            parse_review(r, i, Tokenizer())
    r = raw(i)
    value = json.loads(r['output_text'])
    value['pair']['i2t'] = 'A blue circle.'
    r['output_text'] = json.dumps(value)
    with pytest.raises(ValueError, match='keep changed'):
        parse_review(r, i, Tokenizer())


def test_historical_model_must_be_explicit_and_match_both_model_fields():
    i = item()
    r = raw(i)
    model = 'qwen3.8-max'
    r.update(requested_model=model, returned_model=model)
    with pytest.raises(ValueError):
        parse_review(r, i, Tokenizer())
    assert parse_review(r, i, Tokenizer(), expected_model=model)['review']['decision'] == 'keep'
    for field in ('requested_model', 'returned_model'):
        with pytest.raises(ValueError):
            parse_review({**r, field: 'deepseek-v4.1-flash'}, i, Tokenizer(), expected_model=model)


@pytest.mark.parametrize('model', ['qwen3.8-max', 'deepseek-v4.1-flash'])
def test_all_items_are_reviewed_and_export_matches_raw(tmp_path, model):
    class Client:
        calls = []

        async def generate(self, i, number):
            self.calls.append((i['key'], number))
            return {**raw(i), 'requested_model': model, 'returned_model': model}

    c = config()
    c['sii']['vision_models'] = [model]
    state = ReviewDB(tmp_path, c)
    client = Client()
    try:
        state.admit([item('one'), item('two')])
        result = asyncio.run(review_items(state, client, c, Tokenizer(), asyncio.Event()))
        assert result['counts'] == {'ready': 2}
        assert len(client.calls) == 2  # Existing good text is still sent to SII.
        manifest = export_review(state, Tokenizer())
        assert manifest['records'] == 2 and manifest['codex_calls'] == 0
        assert manifest['model'] == model
        with gzip.open(manifest['output'], 'rt') as f:
            rows = [json.loads(line) for line in f]
            assert len(rows) == 2 and all(r['model'] == model for r in rows)
        state.db.execute("UPDATE items SET result='{}' WHERE key='one'")
        state.db.commit()
        with pytest.raises(ValueError, match='differs'):
            export_review(state, Tokenizer())
    finally:
        state.close()


@pytest.mark.parametrize('model', ['qwen3.8-max', 'deepseek-v4.1-flash'])
def test_interrupted_durable_response_replayed_without_api(tmp_path, model):
    class NeverCall:
        async def generate(self, i, number):
            raise AssertionError('already received')

    c = config()
    c['sii']['vision_models'] = [model]
    state = ReviewDB(tmp_path, c)
    i = item()
    try:
        state.admit([i])
        state.db.execute("UPDATE items SET status='running',attempts=1")
        response = {**raw(i), 'requested_model': model, 'returned_model': model}
        state.db.execute('INSERT INTO attempts VALUES (?,?,?)', ('one', 1, gzip.compress(json.dumps(response).encode())))
        state.db.commit()
        result = asyncio.run(review_items(state, NeverCall(), c, Tokenizer(), asyncio.Event()))
        assert result['counts'] == {'ready': 1}
    finally:
        state.close()


def test_disabled_fallback_is_mandatory(tmp_path):
    c = config()
    c['codex_fallback']['enabled'] = True
    with pytest.raises(ValueError, match='no Codex fallback'):
        asyncio.run(run_farm(SimpleNamespace(root=tmp_path), c, Tokenizer()))


def test_failed_quality_gate_blocks_before_loading_credentials(tmp_path, monkeypatch):
    report = tmp_path / 'qualification.json'
    report.write_text(json.dumps({'status': 'failed', 'visual_review': True}))

    def forbidden():
        raise AssertionError('must not read credentials or connect')

    monkeypatch.setattr('data_synthesis.review_farm.load_sii_settings', forbidden)
    args = SimpleNamespace(root=tmp_path, mode='bulk', qualification=report, selection=tmp_path/'selection.json')
    with pytest.raises(ValueError, match='visual qualification'):
        asyncio.run(run_farm(args, config(), Tokenizer()))


def test_explicit_user_acceptance_defers_semantics_without_fake_pass(tmp_path):
    c = config()
    report = tmp_path / 'acceptance.json'
    args = SimpleNamespace(qualification=report, selection=tmp_path/'selection.json')
    receipt = {'status': 'user_accepted_for_generation', 'authorized_by': 'user',
               'contract': VERSION, 'config': c, 'selection': str(args.selection.resolve()),
               'semantic_quality_passed': False, 'defer_semantic_repair': True,
               'codex_enabled_this_stage': False}
    report.write_text(json.dumps(receipt))
    assert validate_qualification(args, c)['semantic_quality_passed'] is False
    for key, value in [('codex_enabled_this_stage', True), ('semantic_quality_passed', True),
                       ('selection', '/different/selection.json'), ('authorized_by', 'model')]:
        report.write_text(json.dumps({**receipt, key: value}))
        with pytest.raises(ValueError, match='exact SII-only'):
            validate_qualification(args, c)


def test_terminal_failures_are_preserved_not_published_as_success(tmp_path):
    class Bad:
        async def generate(self, i, number):
            return {'status': 'failed', 'error_type': 'content', 'error': 'bad output'}

    c = config()
    c['sii']['max_attempts'] = 1
    state = ReviewDB(tmp_path, c)
    try:
        state.admit([item()])
        result = asyncio.run(review_items(state, Bad(), c, Tokenizer(), asyncio.Event()))
        assert result['state'] == 'needs_attention'
        assert state.db.execute('SELECT count(*) FROM attempts').fetchone()[0] == 1
        report = export_review(state, Tokenizer())
        assert report['state'] == 'needs_attention' and report['outcomes'] == {'failed': 1}
    finally:
        state.close()


def test_direct_mss_is_negotiated_before_connection():
    import socket
    from data_synthesis.direct_backend import DirectMSSBackend

    async def check():
        peer_mss = []

        async def echo(reader, writer):
            peer_mss.append(writer.get_extra_info('socket').getsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG))
            writer.write(await reader.read(100))
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(echo, '127.0.0.1', 0)
        try:
            stream = await DirectMSSBackend(512).connect_tcp('127.0.0.1', server.sockets[0].getsockname()[1], timeout=3)
            await stream.write(b'hello', timeout=3)
            assert await stream.read(100, timeout=3) == b'hello'
            assert peer_mss and 0 < peer_mss[0] <= 512
            await stream.aclose()
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(check())


def test_one_timeout_burst_trips_once_and_late_results_cannot_change_probe():
    api = config()['sii']
    api.update(concurrency_start=256, concurrency_max=512, circuit_failures=5, circuit_seconds=30)
    scheduler = dict(effective=256, healthy=0, failures=0, circuit_until=0, outages=0, generation=0)
    timeout = dict(status='failed', error_type='transport', error='TimeoutError:')
    for _ in range(12):
        update_transport_health(scheduler, api, timeout, generation=0, now=100)
    assert scheduler['outages'] == 1 and scheduler['effective'] == 128
    assert scheduler['circuit_until'] == 130
    before = dict(scheduler)
    update_transport_health(scheduler, api, {'status': 'completed'}, generation=0, now=120)
    assert scheduler == before  # A stale success cannot prematurely reopen the pool.
    update_transport_health(scheduler, api, timeout, generation=1, probe=True, now=130)
    assert scheduler['outages'] == 2 and scheduler['circuit_until'] == 190
    assert scheduler['effective'] == 64
    update_transport_health(scheduler, api, {'status': 'completed'}, generation=2, probe=True, now=190)
    recovered = dict(scheduler)
    update_transport_health(scheduler, api, timeout, generation=0, now=191)
    assert scheduler == recovered and scheduler['circuit_until'] == 0


def test_content_errors_do_not_reduce_transport_capacity_and_growth_is_bounded():
    api = config()['sii']
    api['concurrency_max'] = 16
    scheduler = dict(effective=8, healthy=0, failures=0, circuit_until=0, outages=0, generation=0)
    for _ in range(32):
        update_transport_health(scheduler, api, {'status': 'completed', 'output_text': 'invalid'}, generation=0)
    assert scheduler['effective'] == 16 and scheduler['outages'] == 0


def test_concurrent_failed_requests_preserve_all_evidence_but_trip_once(tmp_path):
    class Client:
        async def generate(self, i, number):
            return {'status': 'failed', 'error_type': 'transport', 'error': 'TimeoutError:'}
    c = config()
    c['sii'].update(concurrency_start=12, concurrency_max=12, max_attempts=1, circuit_failures=5)
    state = ReviewDB(tmp_path, c)
    try:
        state.admit([item(str(n)) for n in range(12)])
        result = asyncio.run(review_items(state, Client(), c, Tokenizer(), asyncio.Event()))
        assert result['counts'] == {'failed': 12}
        assert result['scheduler']['effective'] == 6 and result['scheduler']['outages'] == 1
        assert state.db.execute('SELECT count(*) FROM attempts').fetchone()[0] == 12
    finally:
        state.close()
