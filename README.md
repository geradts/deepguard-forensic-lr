# DeepGuard v10 — Benchmark Runner

DeepGuard v10 connects a forensic deepfake benchmark registry to detector adapters, feature extraction and a LiR hand-off.

## Modern datasets

- DF40 — 40 generation techniques and million-level data.
- AV-Deepfake1M — >1M videos, >2,000 subjects, audio/video/audio-visual manipulation.
- AV-Deepfake1M++ — 2025 extension with roughly 2M clips and perturbations.
- DeepfakeBench — common benchmark/detector framework.
- Deepfake-Eval-2024 — in-the-wild 2024 multimodal evaluation.
- MAVOS-DD — multilingual open-set audio-video evaluation.
- HydraFake — cross-model/cross-forgery/cross-domain evaluation resource.
- GenVidBench — large-scale AI-generated-video benchmark.

## Design principle

Development and external validation are strictly separated. Detector selection, feature selection, fusion and calibration must be frozen before external validation.

Detector outputs are measurements, not LRs. The final evidential interpretation is performed by the LiR layer after development/calibration.

## Run

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Large datasets are not automatically downloaded. The application generates official acquisition commands because access terms and dataset licences differ.

## Repository structure

- `app.py` — Streamlit benchmark runner
- `benchmark.py` — dataset registry and study-plan generator
- `detector_adapters.md` — detector integration contract
- `lir_handoff.yaml` — LiR hand-off template
- `requirements.txt` — Python dependencies

## Reproducibility

Record dataset version/access date, license/EULA, source/subject/generator IDs, detector commit, checkpoint hash, Python environment, FFmpeg/OpenCV versions, feature extractor version, LiR version, calibration parameters and split hashes.

This is a research framework and does not by itself establish forensic validity.