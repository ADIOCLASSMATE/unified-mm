# ImageNet-100 hyperparameter conclusion

## Final default

- Dataset: balanced ImageNet-100 class-conditioned training.
- Training: 80 epochs, 64 Ascend 910B NPUs (`4 x 16`), global batch size 1024.
- Per-rank batch size: 16; gradient accumulation: 1.
- Backbone LR: `30e-5`.
- Special-token LR: `30e-5`.
- Flow-head LR: `4e-5`.
- Projector LR: `4e-5`.

The Backbone/Special and Flow-head/Projector learning rates remain coupled.
This setting is the final hyperparameter choice for subsequent paper experiments.
The exploratory ImageNet-100 executable configs and launchers were retired;
the formal runtime configuration is now the ImageNet-1K 800-epoch contract.

## Evidence

Every point below completed 80 epochs / 8,960 optimizer steps at global batch
1024. FID and IS use the final EMA checkpoint, 10,000 generated samples, 100
Heun sampling steps, CFG 3.5, canonical noise pairing, and the frozen
ImageNet-100 real-stat cache.

| Backbone/Special LR | Flow/Projector LR | FID | IS |
| ---: | ---: | ---: | ---: |
| `20e-5` | `3e-5` | 24.783752970879846 | 69.7525749206543 |
| `20e-5` | `4e-5` | 25.056830839126178 | 69.90226135253906 |
| `20e-5` | `5e-5` | 24.894248666545423 | 70.21249847412109 |
| `24e-5` | `3e-5` | 24.602314272498802 | 70.34276580810547 |
| `24e-5` | `4e-5` | 24.61428706550106 | 69.95282669067383 |
| `24e-5` | `5e-5` | 24.57590606604748 | 70.45716400146485 |
| `30e-5` | `3e-5` | 24.11096738481433 | 70.24663772583008 |
| `30e-5` | `4e-5` | **24.057695924272537** | **71.04673767089844** |
| `30e-5` | `5e-5` | 24.888282026505294 | 70.84093170166015 |

`30e-5 / 4e-5` is best on both criteria: it has the lowest FID and highest IS
in the complete integer 3 x 3 sweep. A planned `36e-5 / 4e-5` boundary check
was stopped before completion and is not part of the evidence or conclusion.

## Historical scaling reference

The preceding global-batch-512 sweep selected Backbone/Special `12e-5` and
Flow/Projector `2e-5` by the balanced FID/IS rule (FID 23.727748879189676, IS
69.72488479614258). Linear batch scaling placed the BS1024 center at
`24e-5 / 4e-5`; the local sweep then selected `30e-5 / 4e-5`.
