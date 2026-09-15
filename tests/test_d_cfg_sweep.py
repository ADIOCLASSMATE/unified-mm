import json

import pytest

from scripts import report_d_cfg_sweep as report


def test_pending_cfg_remains_missing(tmp_path):
    assert report.score(tmp_path, 'missing.json', 1.0, {'id': 'd_on_b'}) is None


@pytest.mark.parametrize('bad_guard', [
    {},
    {'complete': False},
    {'comparison': 'shape_only'},
    {'checked_parameters': []},
])
def test_d_rejects_unverified_time_embeddings(tmp_path, monkeypatch, bad_guard):
    guard = {'schema': 'dynamic_xt_hf_time_embedding_values_v1', 'complete': True,
             'comparison': 'exact_after_model_dtype_cast',
             'checked_parameters': [f'model.backbone_flow_time_embedder.mlp.{i}.{k}'
                                    for i in (0, 2) for k in ('weight', 'bias')]}
    data = {'model_source_load': {'post_load_validation': guard},
            'strategies': {'spatial_halton': {}}}
    path = tmp_path / 'metrics.json'
    path.write_text(json.dumps(data))
    monkeypatch.setattr(report, 'common_score', lambda *args: {'fid': 4.4})
    assert report.score(tmp_path, path.name, 2.0, {'id': 'd_on_b'}) == {'fid': 4.4}
    data['model_source_load']['post_load_validation'] = {**guard, **bad_guard} if bad_guard else {}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='time embedding load'):
        report.score(tmp_path, path.name, 2.0, {'id': 'd_on_b'})
