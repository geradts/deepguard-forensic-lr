from pathlib import Path
import pandas as pd
import streamlit as st
from benchmark import DATASETS, build_plan, validate_manifest

st.set_page_config(page_title='DeepGuard v10', page_icon='◈', layout='wide')
st.title('DeepGuard v10 ◈ Benchmark Runner')
st.write('Modern datasets → detector adapters → feature table → LiR → independent validation.')

tabs = st.tabs(['Study','Datasets','Commands','Feature table','LiR','Audit'])

with tabs[0]:
    st.subheader('Research design')
    dev = st.multiselect('Development', list(DATASETS), default=['DF40','AV-Deepfake1M','DeepfakeBench','MAVOS-DD','HydraFake'])
    ext = st.multiselect('External validation', list(DATASETS), default=['Deepfake-Eval-2024','AV-Deepfake1M++','GenVidBench'])
    if st.button('Build benchmark plan', type='primary'):
        st.session_state['plan'] = build_plan(dev, ext)
    if st.session_state.get('plan'):
        p = st.session_state['plan']
        st.dataframe(pd.DataFrame(p['matrix']), use_container_width=True, hide_index=True)
        st.download_button('Download benchmark plan', p['yaml'], 'deepguard_v10_benchmark.yaml', 'text/yaml')

with tabs[1]:
    st.subheader('Modern benchmark registry')
    rows = []
    for name, d in DATASETS.items():
        rows.append({'dataset':name,'year':d['year'],'role':d['role'],'scale':d['scale'],'modality':d['modality'],'access':d['access'],'source':d['source']})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption('Access-controlled datasets require acceptance of their own terms/EULAs.')

with tabs[2]:
    st.subheader('Acquisition / benchmark commands')
    plan = st.session_state.get('plan')
    if plan:
        for name, cmds in plan['commands'].items():
            with st.expander(name):
                for cmd in cmds: st.code(cmd, shell=True)
    else: st.info('Build the benchmark plan first.')

with tabs[3]:
    st.subheader('Feature table')
    f = st.file_uploader('CSV with detector features', type=['csv'])
    if f:
        df = pd.read_csv(f)
        ok, msg = validate_manifest(df)
        if ok:
            st.success(msg)
            st.dataframe(df.head(40), use_container_width=True, hide_index=True)
            st.download_button('Download feature table', df.to_csv(index=False), 'deepguard_v10_features.csv', 'text/csv')
        else: st.error(msg)

with tabs[4]:
    st.subheader('LiR hand-off')
    st.markdown('Detector scores are measurements, not LRs. Freeze the development/calibration partition before independent external validation.')
    st.code(Path('lir_handoff.yaml').read_text(), language='yaml')

with tabs[5]:
    st.subheader('Audit / reproducibility')
    st.markdown('Record dataset version/access date, licence/EULA, source/subject/generator IDs, detector commit, checkpoint hash, Python environment, FFmpeg/OpenCV versions, feature extractor version, LiR version, calibration parameters, split hashes and evidence SHA-256. External validation must not influence model, feature or calibration selection.')
