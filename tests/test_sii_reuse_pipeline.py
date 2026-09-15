import asyncio
import base64
import copy
import io
import json
import os
from pathlib import Path
import signal
import sqlite3

import httpx
from PIL import Image, ImageDraw
import pytest

from data_synthesis.clients import SIIClient, frozen_pixels
from data_synthesis.config import SIISettings, load_config, load_sii_settings
from data_synthesis.contract import CONTRACT_HASH, parse_pair, text_pair
from data_synthesis.io import dumps, sha
from data_synthesis.pipeline import run
from data_synthesis.publication import audit, export
from data_synthesis.reuse import choose_reuse
from data_synthesis.sources import caption_candidate, freeze, ingest_candidates
from data_synthesis.state import State


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def config():
    c = load_config()
    c["sii"]["vision_models"] = ["qwen3.8-max"]  # Preserve explicit historical fixture provenance.
    c["sii"]["max_attempts"] = 3
    c["codex_fallback"]["enabled"] = True  # Explicit legacy routing contract, independent of SII-only production defaults.
    c["compute_hashes"] = True  # These legacy-contract tests deliberately exercise checksum verification.
    c["minimum_images"] = 2
    c["sii"].update(concurrency_start=2, concurrency_max=4, retry_base_seconds=0.001,
                    retry_max_seconds=0.002, circuit_seconds=0.005, circuit_failures=2,
                    rpm=600000, tpm=600000000)
    c["codex_fallback"]["max_items_per_run"] = 2
    return c


def source(tmp_path, number, *, caption=None, critical=False):
    image = Image.new("RGB", (512, 512), (30 + number * 30, 220 - number * 20, 80))
    ImageDraw.Draw(image).rectangle([20 + number * 20, 50, 90 + number * 23, 270], fill="red")
    output = io.BytesIO()
    image.save(output, format="PNG")
    data = output.getvalue()
    path = tmp_path / f"image-{number}.png"
    path.write_bytes(data)
    row = {"source": "pixmo_cap", "source_id": str(number), "split": "train",
           "capabilities": ["ocr"] if critical else ["relation"]}
    view = {"source_path": str(path), "original_ref": str(path), "view_sha256": sha(data), "source_sha256": sha(data),
            "original_size": [512, 512], "crop": [0, 0, 512, 512], "extension": "png", "view_version": "rgb512-full-frame-pad-v1"}
    if caption is not None:
        row["caption_candidates"] = [caption_candidate(row, caption, author="human", kind="human_caption", provenance={"dataset":"fixture"})]
    return row, view


def prepared(tmp_path, rows):
    root = tmp_path / "prepared"
    root.mkdir()
    db = sqlite3.connect(root / "state.sqlite3")
    db.execute("CREATE TABLE tasks(key TEXT,row TEXT,view TEXT,status TEXT)")
    for i, (row, view) in enumerate(rows):
        db.execute("INSERT INTO tasks VALUES (?,?,?,'prepared')", (str(i), dumps(row), dumps(view)))
    db.commit()
    db.close()
    return root


def pool(tmp_path, rows, c=None):
    c = c or config()
    root = tmp_path / "farm"
    result = freeze(root, c, prepared=[prepared(tmp_path, rows)], pilot=True)
    assert result["state"] == "frozen"
    return root, c


def raw(item, backend, number, *, valid=True):
    pair = text_pair(item["key"], "A red rectangle appears on a green background.")
    prompt = "fixture prompt"
    output = dumps(pair) if valid else '{"truncated":'
    model = "qwen3.8-max" if backend == "sii" else "gpt-5.6-sol"
    result = {"backend": backend, "requested_model": model, "status": "completed", "output_text": output,
              "image_id": item["key"], "view_sha256": item["view"]["view_sha256"], "image_attached": True,
              "contract_hash": CONTRACT_HASH, "attempt": number, "proxy": False,
              "prompt": prompt, "prompt_sha256": sha(prompt.encode())}
    if backend == "codex_fallback":
        result.update(reasoning_effort="low", command=["codex", "exec", "--ephemeral", "--image", "image.png",
                                                      "--model", "gpt-5.6-sol", "-c", 'model_reasoning_effort="low"'],
                      events="\n".join(dumps(e) for e in [
                          {"type":"thread.started","thread_id":"fixture"},
                          {"type":"item.completed","item":{"type":"agent_message","text":output}},
                          {"type":"turn.completed","usage":{"output_tokens":50}}]))
    return result


class FakeClient:
    def __init__(self, backend="sii", failures=0, transport=False, signal_once=False):
        self.backend, self.failures, self.transport = backend, failures, transport
        self.calls, self.active, self.maximum = [], 0, 0
        self.signal_once = signal_once

    async def generate(self, item, number):
        self.calls.append((item["key"], number))
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        if self.signal_once:
            self.signal_once = False
            os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.sleep(0.005)
        self.active -= 1
        result = raw(item, self.backend, number, valid=number > self.failures)
        if self.transport and number <= self.failures:
            result.update(status="failed", error_type="transport", error="fixture outage")
        return result

    async def close(self):
        pass


def test_shell_settings_ignore_examples_and_do_not_execute_rc(tmp_path):
    marker = tmp_path / "executed"
    rc = tmp_path / ".zshrc"
    rc.write_text(f'touch {marker}\nexport SII_API_KEY="secret-key"\nexport SII_BASE_URL="https://sii.example.invalid/v1"\n')
    settings = load_sii_settings(environ={}, rc_paths=[rc])
    assert settings.api_key == "secret-key" and settings.endpoint("openai_chat").endswith('/v1/chat/completions')
    assert not marker.exists() and 'secret-key' not in repr(settings) + dumps(settings.public())
    rc.write_text(f'export SII_API_KEY="$(touch {marker})"\nexport SII_BASE_URL="https://sii.example.invalid"\n')
    with pytest.raises(ValueError, match="literal"):
        load_sii_settings(environ={}, rc_paths=[rc])
    assert not marker.exists()
    with pytest.raises(ValueError, match="SII_BASE_URL"):
        load_sii_settings(environ={"SII_API_KEY":"key"}, rc_paths=[])
    explicit = load_sii_settings(environ={"SII_API_KEY":"env-key","SII_BASE_URL":"https://env.example.invalid"}, rc_paths=[rc])
    assert explicit.api_key == "env-key"


def test_conflicting_shell_exports_require_explicit_environment(tmp_path):
    one, two = tmp_path/'bashrc', tmp_path/'zshrc'
    one.write_text('export SII_BASE_URL="https://one.example.invalid"\n')
    two.write_text('export SII_BASE_URL="https://two.example.invalid"\n')
    with pytest.raises(ValueError, match="conflicting"):
        load_sii_settings(environ={"SII_API_KEY":"key"}, rc_paths=[one,two])


def test_short_caption_reuse_has_no_api_or_codex_call_and_no_word_floor(tmp_path):
    root, c = pool(tmp_path, [source(tmp_path,0,caption="A red rectangle."),source(tmp_path,1,caption="Another red rectangle.")])
    sii, codex = FakeClient(), FakeClient("codex_fallback")
    result = asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=sii,codex=codex))
    assert result['state']=='completed' and not sii.calls and not codex.calls
    assert result['run_counters']['reuse']==2
    with State(root,readonly=True) as state:
        assert state.db.execute('SELECT count(*) FROM attempts').fetchone()[0]==0


def test_no_hash_pool_reuses_and_publishes_by_explicit_image_reference(tmp_path, monkeypatch):
    c = config()
    c["compute_hashes"] = False
    a, b = source(tmp_path, 0, caption="A red rectangle."), source(tmp_path, 1)
    # Equal pixels under different source IDs/references are deliberately not content-deduplicated.
    Path(b[1]["source_path"]).write_bytes(Path(a[1]["source_path"]).read_bytes())
    duplicate = copy.deepcopy(a)
    duplicate[0]["source_id"] = "alias"
    a[0]["url"] = duplicate[0]["url"] = "https://example.invalid/same-original"

    def no_digest(*args, **kwargs):
        raise AssertionError("content/file hashing must remain disabled")

    for module in ("sources", "publication"):
        monkeypatch.setattr(f"data_synthesis.{module}.file_sha", no_digest)
    for module in ("clients", "reuse", "state", "publication"):
        monkeypatch.setattr(f"data_synthesis.{module}.sha", no_digest)
    root, c = pool(tmp_path, [a, b, duplicate], c)
    with State(root, readonly=True) as state:
        assert state.meta("frozen")["images"] == 2
        assert state.db.execute("SELECT count(*) FROM items WHERE view_sha256 IS NULL AND source_sha256 IS NULL").fetchone()[0] == 2

    class ReferenceClient(FakeClient):
        async def generate(self, item, number):
            result = await super().generate(item, number)
            result.update(view_id=item["view"]["view_id"], prompt_sha256=None)
            return result

    sii, codex = ReferenceClient(), FakeClient("codex_fallback")
    result = asyncio.run(run(root, c, tokenizer=Tokenizer(), sii=sii, codex=codex))
    assert result["state"] == "completed" and len(sii.calls) == 1 and not codex.calls
    destination = tmp_path / "release"
    report = export(root, destination, tokenizer=Tokenizer())
    assert report["verified_images"] == 2 and report["compute_hashes"] is False
    assert report["manifest_sha256"] is None and not report["all_source_and_view_sha256_verified"]
    assert report["all_image_references_verified"]
    assert json.loads((destination / "publication.json").read_text())["file_sha256"] == {}
    with State(root) as state:
        attempt = state.db.execute("SELECT id,raw_sha256,result_sha256 FROM attempts").fetchone()
        assert attempt["raw_sha256"] is None and attempt["result_sha256"] is None
        response = state.raw(attempt["id"])
        response["view_id"] = "wrong-image-reference"
        import gzip
        state.db.execute("UPDATE attempts SET raw_gzip=? WHERE id=?",
                         (gzip.compress(dumps(response).encode()), attempt["id"]))
        state.db.commit()
    with pytest.raises(ValueError, match="attachment provenance"):
        audit(destination, tokenizer=Tokenizer())


def test_no_hash_annotation_requires_reference_binding(tmp_path):
    c = config()
    c["compute_hashes"] = False
    row, view = source(tmp_path, 0)
    view.update(view_sha256=None, source_sha256=None, hashes_computed=False, view_id=view["source_path"])
    fact = {"type": "count", "entity": "red squares", "count": 3, "verified": True,
            "view_sha256": None, "fully_visible": True, "exhaustive_for_referent": True,
            "provenance": {"annotator": "human"}}
    row["verified_facts"] = [fact]
    item = {"key": "item", "identity": "pixmo_cap:0", "row": row, "view": view}
    assert choose_reuse(item, Tokenizer(), c)[0] is None
    fact["view_id"] = view["view_id"]
    assert choose_reuse(item, Tokenizer(), c)[1]["route"] == "annotation"


def test_crop_or_unverified_ocr_does_not_blindly_reuse_caption(tmp_path):
    row, view = source(tmp_path,0,caption='A sign says EXIT.',critical=True)
    item={'key':view['source_sha256'],'identity':'pixmo_cap:0','row':row,'view':view}
    assert choose_reuse(item,Tokenizer(),config())[0] is None
    row['readability_view_sha256']=view['view_sha256']
    assert choose_reuse(item,Tokenizer(),config())[0]['i2t']=='A sign says EXIT.'
    view['original_size']=[700,512]
    assert choose_reuse(item,Tokenizer(),config())[0] is None


def test_verified_annotation_rendering_is_explicit_and_has_no_guessed_zero(tmp_path):
    row, view=source(tmp_path,0)
    fact={'type':'count','entity':'red squares','count':3,'verified':True,'view_sha256':view['view_sha256'],
          'fully_visible':True,'exhaustive_for_referent':True,'provenance':{'annotator':'human'}}
    row['verified_facts']=[fact]
    item={'key':view['source_sha256'],'identity':'pixmo_cap:0','row':row,'view':view}
    pair,evidence,_=choose_reuse(item,Tokenizer(),config())
    assert evidence['route']=='annotation' and pair['observations']['counts'][0]['count']==3
    fact['count']=0
    assert choose_reuse(item,Tokenizer(),config())[0] is None


def test_valid_sii_result_is_final_without_codex_review(tmp_path):
    root,c=pool(tmp_path,[source(tmp_path,0)])
    sii,codex=FakeClient(),FakeClient('codex_fallback')
    result=asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=sii,codex=codex))
    assert result['state']=='completed' and len(sii.calls)==1 and not codex.calls
    assert result['run_counters']['sii_accepted']==1


def test_sii_quality_retries_then_codex_once_with_complete_history(tmp_path):
    root,c=pool(tmp_path,[source(tmp_path,0)])
    sii,codex=FakeClient(failures=99),FakeClient('codex_fallback')
    result=asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=sii,codex=codex))
    assert result['state']=='completed'
    assert [n for _,n in sii.calls]==[1,2,3] and len(codex.calls)==1
    with State(root,readonly=True) as state:
        attempts=state.db.execute('SELECT backend,number,status FROM attempts ORDER BY started_at').fetchall()
        assert [tuple(row) for row in attempts]==[('sii',1,'failed'),('sii',2,'failed'),('sii',3,'failed'),('codex_fallback',1,'succeeded')]
    release=tmp_path/'release'
    assert export(root,release,tokenizer=Tokenizer())['all_codex_calls_follow_sii_exhaustion']


def test_received_response_is_replayed_after_crash_without_spending_again(tmp_path):
    root,c=pool(tmp_path,[source(tmp_path,0)])
    with State(root,c) as state:
        key=state.db.execute('SELECT key FROM items').fetchone()[0]
        ident,number=state.claim(key,'sii')
        state.receive(ident,raw(state.item(key),'sii',number))
    sii,codex=FakeClient(),FakeClient('codex_fallback')
    result=asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=sii,codex=codex))
    assert result['state']=='completed' and not sii.calls and not codex.calls


def test_failed_api_attempts_survive_restart_before_codex(tmp_path):
    root,c=pool(tmp_path,[source(tmp_path,0)])
    with State(root,c) as state:
        key=state.db.execute('SELECT key FROM items').fetchone()[0]
        for _ in range(2):
            ident,n=state.claim(key,'sii')
            state.receive(ident,raw(state.item(key),'sii',n,valid=False))
            state.fail(key,'retry','bad json',ident=ident)
    sii,codex=FakeClient(failures=99),FakeClient('codex_fallback')
    result=asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=sii,codex=codex))
    assert result['state']=='completed' and [n for _,n in sii.calls]==[3] and len(codex.calls)==1


def test_codex_failure_is_quarantined_and_not_silently_published(tmp_path):
    root,c=pool(tmp_path,[source(tmp_path,0)])
    result=asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=FakeClient(failures=99),codex=FakeClient('codex_fallback',failures=99)))
    assert result['counts']=={'failed':1}
    with pytest.raises(ValueError,match='pending, failed'):
        export(root,tmp_path/'release',tokenizer=Tokenizer())
    sii,codex=FakeClient(),FakeClient('codex_fallback')
    asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=sii,codex=codex))
    assert not sii.calls and not codex.calls


def test_transport_circuit_recovers_and_does_not_make_sol_the_default(tmp_path):
    root,c=pool(tmp_path,[source(tmp_path,i) for i in range(4)])
    sii,codex=FakeClient(failures=1,transport=True),FakeClient('codex_fallback')
    result=asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=sii,codex=codex))
    assert result['state']=='completed' and not codex.calls and sii.maximum<=c['sii']['concurrency_max']


def test_interrupt_drains_received_work_and_resume_keeps_success(tmp_path):
    root,c=pool(tmp_path,[source(tmp_path,i) for i in range(4)])
    sii=FakeClient(signal_once=True)
    result=asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=sii,codex=FakeClient('codex_fallback')))
    assert result['state']=='interrupted'
    done=set(k for k,n in sii.calls)
    second=FakeClient()
    assert asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=second,codex=FakeClient('codex_fallback')))['state']=='completed'
    assert not (done & {k for k,n in second.calls})


def test_freeze_deduplicates_images_and_keeps_source_captions(tmp_path):
    one=source(tmp_path,0,caption='Caption one.')
    two=copy.deepcopy(one)
    two[0]['source_id']='alias'
    two[0]['caption_candidates']=[caption_candidate(two[0],'Caption two.',author='human',kind='human_caption',provenance={'source':'other'})]
    root,c=pool(tmp_path,[one,two,source(tmp_path,1,caption='Different color.')])
    with State(root,readonly=True) as state:
        assert state.meta('frozen')['records']==2
        item=state.item(one[1]['source_sha256'])
        assert len(item['row']['caption_candidates'])==2
    assert asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=FakeClient(),codex=FakeClient('codex_fallback')))['state']=='completed'


def test_production_run_requires_qualification_and_non_capped_minimum(tmp_path):
    c=config()
    root,c=pool(tmp_path,[source(tmp_path,0)],c)
    with State(root,c) as state:
        value=state.meta('frozen');value['pilot']=False
        state.set_meta('frozen',value);state.db.commit()
    with pytest.raises(ValueError,match='qualification'):
        asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=FakeClient()))
    changed=copy.deepcopy(c);changed['sii']['max_attempts']=1
    with pytest.raises(ValueError,match='contract changed'):
        State(root,changed)


def test_sharded_publication_and_raw_audit_tamper_detection(tmp_path):
    root,c=pool(tmp_path,[source(tmp_path,0,caption='A rectangle.'),source(tmp_path,1)])
    asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=FakeClient(),codex=FakeClient('codex_fallback')))
    release=tmp_path/'release'
    report=export(root,release,tokenizer=Tokenizer(),shard_records=1)
    assert report['routes']=={'reuse':1,'sii':1}
    manifest=json.loads((release/'text_index.json').read_text())
    assert len(manifest['t2i']['shards'])==2
    assert audit(release,tokenizer=Tokenizer())['verified_images']==2
    with State(root) as state:
        state.db.execute("UPDATE attempts SET raw_sha256='broken'")
        state.db.commit()
    with pytest.raises(ValueError,match='checksum'):
        audit(release,tokenizer=Tokenizer())


def test_direct_sii_payload_matches_frozen_pixels_and_scrubs_key(tmp_path,monkeypatch):
    row,view=source(tmp_path,0)
    item={'key':view['source_sha256'],'identity':'pixmo_cap:0','row':row,'view':view}
    for name in ('HTTP_PROXY','http_proxy','HTTPS_PROXY','ALL_PROXY'):
        monkeypatch.setenv(name,'http://127.0.0.1:1')
    before=dict(os.environ)
    requests=[]
    async def handle(request):
        body=json.loads(request.content)
        requests.append(body)
        assert str(request.url)=='https://sii.example.invalid/v1/chat/completions'
        assert request.headers['Authorization']=='Bearer test-private-key'
        url=body['messages'][0]['content'][1]['image_url']['url']
        assert sha(base64.b64decode(url.split(',',1)[1]))==view['view_sha256']
        return httpx.Response(200,json={'model':'qwen3.8-max','choices':[{'finish_reason':'stop','message':{'content':dumps(text_pair(item['key'],'A rectangle.'))}}]})
    async def scenario():
        client=SIIClient(SIISettings('https://sii.example.invalid/','test-private-key'),config()['sii'],transport=httpx.MockTransport(handle))
        try:return await client.generate(item,1)
        finally:await client.close()
    result=asyncio.run(scenario())
    assert result['status']=='completed' and len(requests)==1
    assert 'test-private-key' not in dumps(result) and os.environ==before
    assert parse_pair(result,item['key'],Tokenizer())['i2t']=='A rectangle.'


def test_ingest_preserves_existing_caption_and_never_uses_source_target_as_cap(tmp_path):
    manifest=tmp_path/'input.jsonl'
    rows=[]
    for i in range(3):
        row,view=source(tmp_path,i,caption='Existing caption.')
        row['local_path']=view['source_path'];rows.append(row)
    manifest.write_text(''.join(dumps(r)+'\n' for r in rows))
    result=ingest_candidates(manifest,tmp_path/'supply','source')
    assert result['records']==3
    saved=[json.loads(line) for line in (tmp_path/'supply/candidates/source-00000.jsonl').read_text().splitlines()]
    assert saved[0]['caption_candidates'][0]['text']=='Existing caption.'


def test_repeated_annotation_rows_join_before_download_and_seal_is_immutable(tmp_path):
    from data_synthesis.sources import seal_candidates
    row, view = source(tmp_path, 0, caption='Original caption.')
    row['local_path'] = view['source_path']
    second = copy.deepcopy(row)
    second['caption_candidates'][0]['text'] = 'Another source description.'
    second['annotations'] = [{'entity': 'rectangles', 'points': [[0.1, 0.2]]}]
    manifest = tmp_path/'rows.jsonl'
    manifest.write_text(dumps(row)+'\n'+dumps(second)+'\n')
    supply = tmp_path/'supply'
    assert ingest_candidates(manifest,supply,'points')['records'] == 1
    path = supply/'candidates/points-00000.jsonl'
    joined = json.loads(path.read_text())
    assert len(joined['caption_candidates']) == 2 and joined['annotations'] == second['annotations']
    marker = seal_candidates(supply)
    assert seal_candidates(supply) == marker and marker['records'] == 1
    with pytest.raises(ValueError,match='closed'):
        ingest_candidates(manifest,supply,'late')
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):
        seal_candidates(supply)


def test_view_hash_is_not_a_license_to_reuse_unchecked_generation_prompts(tmp_path):
    row, view = source(tmp_path,0,caption='Ten blue rectangles in watercolor.')
    row['caption_candidates'][0].update(kind='generation_prompt',view_sha256=view['view_sha256'])
    item = {'key':view['source_sha256'],'identity':'pixmo_cap:0','row':row,'view':view}
    assert choose_reuse(item,Tokenizer(),config())[0] is None


def test_test_prompt_exclusion_survives_case_and_whitespace_changes(tmp_path):
    rows = [source(tmp_path,0,caption='A red rectangle.'),source(tmp_path,1,caption='A different rectangle.')]
    prompts = tmp_path/'test-prompts.jsonl'
    prompts.write_text(dumps({'prompt':' A   RED rectangle. '})+'\n')
    root = tmp_path/'farm'
    frozen = freeze(root,config(),prepared=[prepared(tmp_path,rows)],exclude_prompts=prompts,pilot=True)
    assert frozen['records'] == 1 and frozen['exclusions']['evaluation_test_prompt_overlap'] == 1


def test_explicit_terminal_quarantine_preserves_failure_history_and_release_scope(tmp_path):
    rows = [source(tmp_path,0,caption='A rectangle.'), source(tmp_path,1)]
    root,c = pool(tmp_path,rows)
    asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=FakeClient(failures=99),codex=FakeClient('codex_fallback',failures=99)))
    with State(root,c) as state:
        assert state.quarantine_failed(c)['quarantined'] == 1
        assert state.db.execute('SELECT count(*) FROM attempts').fetchone()[0] == 4
    release = tmp_path/'release'
    assert export(root,release,tokenizer=Tokenizer())['records'] == 1
    assert json.loads((release/'publication.json').read_text())['quarantined_images'] == 1


def test_modern_frozen_pool_can_encode_before_text_is_ready(tmp_path):
    from scripts.prepare_b512_posterior_bank import prepare_bank
    root,c = pool(tmp_path,[source(tmp_path,0)])
    bank = tmp_path/'bank'
    receipt = prepare_bank(root,bank)
    assert receipt['records'] == 1 and prepare_bank(root,bank) == receipt
    row = json.loads((bank/'manifest.jsonl').read_text())
    with State(root,readonly=True) as state:
        assert row['key'] == state.db.execute('SELECT key FROM items').fetchone()[0]
        assert state.counts() == {'pending':1}


def test_archive_intake_requires_selected_scope_and_cannot_expand_on_resume(tmp_path):
    from scripts.download_b512_corners_v3 import init_catalogue
    root = tmp_path/'archive'
    with pytest.raises(ValueError,match='provide --catalogue'):
        init_catalogue(root)
    source_file = tmp_path/'upstream.bin';source_file.write_bytes(b'0123')
    path = tmp_path/'selected.json'
    catalogue = {'files':[{'id':'fixture:one','path':str(source_file),'bytes':4,'revision':'fixture-pinned','reuse_existing':True,'url':None}], 'declared_bytes':4}
    path.write_text(dumps(catalogue))
    assert init_catalogue(root,path) == catalogue
    changed = copy.deepcopy(catalogue);changed['files'][0]['bytes']=5;changed['declared_bytes']=5
    path.write_text(dumps(changed))
    with pytest.raises(ValueError,match='changed'):
        init_catalogue(root,path)


def test_export_uses_actual_training_loader_image_identity_not_state_key(tmp_path):
    import torch
    from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset,POSTERIOR_CACHE_FORMAT,POSTERIOR_STATS_LAYOUT
    from data_synthesis.sources import release_rows
    class IntTokenizer:
        eos_token_id=14
        def encode(self,text,add_special_tokens=False):
            return [100+i for i,_ in enumerate(text.split())]
    root,c=pool(tmp_path,[source(tmp_path,0,caption='A red rectangle.')])
    tokenizer=IntTokenizer()
    asyncio.run(run(root,c,tokenizer=tokenizer,sii=FakeClient(),codex=FakeClient('codex_fallback')))
    release=tmp_path/'release';export(root,release,tokenizer=tokenizer)
    means=torch.zeros((1,1024,16),dtype=torch.float16);stds=torch.ones_like(means)
    cache=tmp_path/'posterior.pt'
    torch.save({'posterior_stats':torch.cat((means,stds),dim=-1),'img_ids':torch.tensor([1]),
                'metadata':{'format':POSTERIOR_CACHE_FORMAT,'stats_layout':POSTERIOR_STATS_LAYOUT}},cache)
    samples={}
    for mode in ('i2t','t2i'):
        dataset=ImageNetFlowCacheDataset(cache_path=str(cache),tokenizer=tokenizer,boi_token_id=11,eoi_token_id=12,
            mask_token_id=13,eos_token_id=14,image_tokens_per_img=1024,image_latent_dim=16,
            manifest_jsonl=str(release/'manifest.jsonl'),conditioning_mode='caption',caption_jsonl=str(release/'captions.jsonl'),
            caption_list_key='captions',caption_include_original=True,synthetic_text_index_manifest=str(release/'text_index.json'),
            caption_sequence_modes=[mode],seed=2,max_seq_length=2048)
        samples[mode]=dataset[0]
        assert samples[mode]['image_latents'].shape==(1024,16)
    assert samples['t2i']['image_loss_mask'].sum()==1024
    assert not samples['i2t']['image_loss_mask'].any()
    # New publications can themselves be reused without losing author or identity.
    imported=list(release_rows(release))
    assert len(imported)==1 and imported[0][0]['caption_candidates'][0]['author']=='human'


def test_new_supply_cohort_preserves_previously_frozen_image_bytes(tmp_path):
    import argparse
    from data_synthesis.sources import seal_candidates
    from scripts.supply_b512_images import download_service,prepare_service
    from utils.image_near_duplicates import write_index
    from utils.image_shard_io import read_image_bytes
    images=tmp_path/'shared-images';exclude=tmp_path/'exclude.txt';exclude.write_text('')
    near=tmp_path/'benchmark';write_index(near,[],[],[])
    old_refs=[]
    for number in (0,1):
        row,view=source(tmp_path,number,caption='A rectangle.')
        row['local_path']=view['source_path']
        source_root=tmp_path/f'supply-{number}';prep_root=tmp_path/f'prepared-{number}'
        manifest=tmp_path/f'candidates-{number}.jsonl';manifest.write_text(dumps(row)+'\n')
        ingest_candidates(manifest,source_root,'fixture');seal_candidates(source_root)
        args=argparse.Namespace(root=str(source_root),image_root=str(images),farm_root=str(prep_root),
            workers=1,per_host=1,exclude=str(exclude),near_exclude_index=str(near))
        async def prepare_cohort():
            await asyncio.wait_for(asyncio.gather(download_service(args),prepare_service(args)),timeout=30)
        asyncio.run(prepare_cohort())
        marker=next((source_root/'prepared_batches').glob('*/batch.json'))
        db=sqlite3.connect(marker.parent/'state.sqlite3')
        actual=json.loads(db.execute("SELECT view FROM tasks WHERE status='prepared'").fetchone()[0]);db.close()
        old_refs.append((actual['source_path'],actual['view_sha256']))
        assert all(sha(read_image_bytes(path))==digest for path,digest in old_refs)
    assert old_refs[0][0]!=old_refs[1][0]


def test_generated_test_prompt_is_rejected_even_without_a_source_caption(tmp_path):
    prompts=tmp_path/'evaluation-prompts.txt'
    prompts.write_text('A red rectangle appears on a green background.\n')
    root=tmp_path/'farm';c=config()
    freeze(root,c,prepared=[prepared(tmp_path,[source(tmp_path,0)])],exclude_prompts=prompts,pilot=True)
    result=asyncio.run(run(root,c,tokenizer=Tokenizer(),sii=FakeClient(),codex=FakeClient('codex_fallback')))
    assert result['counts']=={'failed':1}
    with State(root,readonly=True) as state:
        errors=[row[0] for row in state.db.execute('SELECT error FROM attempts')]
        assert len(errors)==4 and all('excluded evaluation test prompt' in e for e in errors)
