# DP-Net

PyTorch implementation of the DP-Net architecture for zero-calibration cross-subject SSVEP decoding under short time windows.

## Included components

- Data-Driven Neural Path (DDNP)
  - gated embedding
  - cross-electrode spatial filtering
  - multi-scale temporal modeling
  - temporal self-attention
  - time-step-wise feature head
- Template-Guided Analytical Path (TGAP)
  - standard harmonic templates
  - source-domain empirical templates
  - QR/SVD template-similarity scores
  - circular temporal shifts and shift penalties
- Analytical Score Preservation Module (ASPM)
- Direct feature concatenation and joint classification head


## Package interface

```python
from models import DPNet, build_empirical_templates

model = DPNet(
    num_classes=40,
    num_electrodes=9,
    time_points=150,
    sampling_rate=250,
)

empirical_templates = build_empirical_templates(
    source_train_eeg,
    source_train_labels,
    frequencies=[8.0 + 0.2 * index for index in range(40)],
    sampling_rate=250,
)
model.set_empirical_templates(empirical_templates)
```

The input tensor to `DPNet` has shape `(batch, electrodes, time)`.

## Dependencies

- Python 3.10 or later
- PyTorch
- NumPy

Install the required packages with:

```bash
pip install -r requirements.txt
```
