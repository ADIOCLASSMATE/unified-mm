import json

import pytest

from scripts.report_qualitative_speed import summarize


def test_paired_speed_uses_per_device_medians_and_rejects_different_inputs(tmp_path):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    contract = {'warmup_batches': 1, 'measured_repeats': 3}
    write(tmp_path / 'manifest.json', {'models': [{'id': m, 'label': m} for m in ('b_x0', 's2_single')],
        'contract': {'cfg': 3.5, 'model_cfg': {'b_x0': 2., 's2_single': 1.5}, 'timing': contract}})
    write(tmp_path / 'COMPLETED.json', {'complete': True})
    for mid, cfg, seconds in [('b_x0', 2., 8.), ('s2_single', 1.5, 4.)]:
        for rank in range(16):
            write(tmp_path / f'models/{mid}/timing/rank-{rank:02d}.json', {
                'rank': rank, 'samples': [{'id': f'paired-{rank}', 'seed': 42}], 'batch_size': 8,
                'world_size': 16, 'generation_seconds': [seconds, seconds * 10, seconds],
                'timing_contract': contract, 'cfg': cfg, 'vae_decode_seconds': 1., 'png_save_seconds': .5,
                'trace': {'backbone_kv_cache_enabled': mid == 'b_x0'}})
    result = summarize(tmp_path)
    assert result['s2_over_b_device_throughput'] == 2.
    assert result['models']['s2_single']['median_device_images_per_second'] == 2.
    assert result['models']['s2_single']['mean_seconds_per_image'] == 2.
    assert result['models']['s2_single']['median_seconds_per_image'] == .5
    assert len((tmp_path / 'speed.csv').read_text().splitlines()) == 97
    path = tmp_path / 'models/s2_single/timing/rank-00.json'
    row = json.loads(path.read_text())
    row['samples'][0]['seed'] = 43
    write(path, row)
    with pytest.raises(ValueError, match='paired'):
        summarize(tmp_path)
