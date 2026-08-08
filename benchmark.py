from __future__ import annotations
import yaml
import pandas as pd

DATASETS={
"DF40":{"year":2024,"role":"development","modality":"image/video","scale":"million-level; 40 generation techniques","access":"research / CC BY-NC 4.0; dataset terms apply","source":"https://github.com/YZY-stack/DF40","commands":["git clone https://github.com/YZY-stack/DF40.git external/DF40"]},
"AV-Deepfake1M":{"year":2024,"role":"development","modality":"audio/video/AV","scale":">1M videos; >2K subjects","access":"EULA required","source":"https://github.com/ControlNet/AV-Deepfake1M","commands":["git clone https://github.com/ControlNet/AV-Deepfake1M.git external/AV-Deepfake1M"]},
"AV-Deepfake1M++":{"year":2025,"role":"external / perturbation","modality":"audio/video/AV","scale":"2M video clips","access":"research-only terms / challenge access","source":"https://deepfakes1m.github.io/2025","commands":["git clone https://github.com/ControlNet/AV-Deepfake1M.git external/AV-Deepfake1M"]},
"DeepfakeBench":{"year":2023,"role":"development framework","modality":"image/video","scale":"multi-dataset benchmark","access":"dataset-specific terms","source":"https://github.com/SCLBD/DeepfakeBench","commands":["git clone https://github.com/SCLBD/DeepfakeBench.git external/DeepfakeBench"]},
"Deepfake-Eval-2024":{"year":2025,"role":"external / in-the-wild","modality":"video/audio/image","scale":"44h video; 56.5h audio; 1,975 images; 88 websites; 52 languages","access":"research dataset terms","source":"https://github.com/nuriachandra/deepfake-eval-2024","commands":["git clone https://github.com/nuriachandra/deepfake-eval-2024.git external/Deepfake-Eval-2024"]},
"MAVOS-DD":{"year":2025,"role":"external / open-set","modality":"multilingual audio/video","scale":">250h; 8 languages; 7 generation models per language","access":"Hugging Face dataset terms","source":"https://huggingface.co/datasets/unibuc-cs/MAVOS-DD","commands":["git lfs install","git clone https://huggingface.co/datasets/unibuc-cs/MAVOS-DD external/MAVOS-DD"]},
"HydraFake":{"year":2026,"role":"external / cross-forgery","modality":"image/face","scale":"52K evaluation images","access":"research model/data terms","source":"https://github.com/EricTan7/Veritas","commands":["git clone https://github.com/EricTan7/Veritas.git external/Veritas"]},
"GenVidBench":{"year":2026,"role":"external / AI-video","modality":"generated video","scale":"6.78M videos","access":"Hugging Face / dataset terms","source":"https://github.com/genvidbench/GenVidBench","commands":["git clone https://github.com/genvidbench/GenVidBench.git external/GenVidBench"]}}

def build_plan(dev,ext):
    matrix=[]; commands={}
    for d in dev:
        for feature_set in ["spatial","spatial_temporal","visual","visual_face","visual_av","full_multimodal"]:
            matrix.append({"dataset":d,"role":"development","feature_set":feature_set})
        commands[d]=DATASETS[d]["commands"]
    for d in ext:
        matrix.append({"dataset":d,"role":"external_validation","feature_set":"frozen_full_multimodal"})
        commands[d]=DATASETS[d]["commands"]
    config={"development":dev,"external_validation":ext,"matrix":matrix,"rules":["freeze model before external validation","group by source/subject/generator","record dataset/model hashes"]}
    return {"matrix":matrix,"commands":commands,"yaml":yaml.safe_dump(config,sort_keys=False)}

def validate_manifest(df):
    missing={"label","source_id"}-set(df.columns)
    if missing:return False,"Missing: "+", ".join(sorted(missing))
    labels=set(pd.to_numeric(df["label"],errors="coerce").dropna().astype(int))
    if labels!={0,1}:return False,"label must contain 0 and 1."
    return True,f"{len(df):,} observations loaded."
