# STEP BY STEP

## CREATE ENVIRONMENT

### KALDI

- create environment `kaldi`

- install `python==2.7.18` and other dependencies (such as `mkl-level` ) with `conda`

- install kaldi

### EXKALDI

- create environment `exkaldi`

- install `python==3.9.13` with `conda`

- install `libboost==1.65.1` with `conda`

- install `tensorflow`, `python-flatbuffers`, `tensorboard` with conda (dependencies of tutorials)

- install `kenlm` with pip (`pip install https://github.com/khanhi2r/kenlm/archive/refs/tags/exkaldi.zip` or `pip install kenlm-exkaldi.zip`)

- install `exkaldi` with pip (`pip install https://github.com/khanhi2r/exkaldi/archive/refs/tags/v1.3.6.zip` or `pip install exkaldi-1.3.6.zip`)

- `chmod +x /home/khanh/workspace/miniconda3/envs/exkaldi/exkaldisrc/tools/lmplz`