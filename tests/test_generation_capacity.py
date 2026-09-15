import pytest
import json

from scripts.benchmark_generation_capacity import bounds, refine_batches, report


def test_capacity_search_requires_real_oom_and_refines_within_bracket():
    rows=[{'batch_size':128,'status':'ok'},{'batch_size':256,'status':'ok'},
          {'batch_size':384,'status':'oom'},{'batch_size':512,'status':'oom'}]
    assert bounds(rows)==(256,384)
    candidates=refine_batches(*bounds(rows))
    assert len(candidates)<=8
    assert all(256<x<384 and x%8==0 for x in candidates)
    assert refine_batches(376,384)==[]


def test_capacity_search_does_not_treat_unknown_errors_as_oom():
    with pytest.raises(RuntimeError,match='non-OOM'):
        bounds([{'batch_size':128,'status':'error'}])
    with pytest.raises(RuntimeError,match='nonmonotonic'):
        bounds([{'batch_size':128,'status':'oom'},{'batch_size':256,'status':'ok'}])


def test_final_report_compares_each_models_own_stable_batch(tmp_path):
    models={}
    for mid,batch,seconds in [('b_x0',256,128),('s2_single',512,64)]:
        models[mid]=[dict(device=d,batch_size=batch,status='ok',warmup_batches=1,
            generation_seconds=[seconds]*3,cfg=2. if mid=='b_x0' else 1.5,
            peak_allocated_bytes=58*1024**3,peak_reserved_bytes=60*1024**3,
            device_total_memory_bytes=61*1024**3,
            trace={'backbone_kv_cache_enabled':mid=='b_x0'}) for d in range(16)]
    (tmp_path/'measurements.json').write_text(json.dumps(dict(complete=True,models=models,
        search={mid:{'first_oom_batch':rows[0]['batch_size']+8} for mid,rows in models.items()},protocol={})))
    report(tmp_path)
    result=json.loads((tmp_path/'summary.json').read_text())
    assert result['s2_over_b_throughput']==4.
    assert result['models']['b_x0']['mean_seconds_per_image']==.5
    assert len((tmp_path/'timings.csv').read_text().splitlines())==97
