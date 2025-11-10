# LocalFlowMatching
Official implementation of "Local Flow Matching Generative Models".

# Environment
```
conda env create -f environment.yml
```

# 2d
```
conda activate lfm
# Rose
python main_2d.py --hyper_param_config config/2d_LFM_rose.yaml
# Tree
python main_2d.py --hyper_param_config config/2d_LFM_tree.yaml
```

# Tabular

First follow https://github.com/gpapamak/maf?tab=readme-ov-file#how-to-get-the-datasets regarding data downloading

The unzipped `data` folder should be structured as `{TASK}/file`:
- BSDS300/BSDS300.hdf5
- gas/ethylene_CO.pickle
- miniboone/data.npy
- power/data.npy

```
conda activate lfm
# Power
python main_tabular.py --hyper_param_config config/tabular_LFM_power.yaml
# Gas
python main_tabular.py --hyper_param_config config/tabular_LFM_gas.yaml
# Miniboone
python main_tabular.py --hyper_param_config config/tabular_LFM_miniboone.yaml
# Bsds300
python main_tabular.py --hyper_param_config config/tabular_LFM_bsds300.yaml
```
