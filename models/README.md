# Model source

- [5DfCF.py](5DfCF.py) contains the 5DfCF model components.
- [Crossformer.py](Crossformer.py) contains the CrossFormer backbone and its attention components.
- [build.py](build.py) contains the model factory used with the [YAML presets](../configs/README.md).
- [utils/data_preprocess_and_load](utils/data_preprocess_and_load/) contains the dataset loader and tensor preparation. The [root README](../README.md#usage) describes the workflow from CSV metadata preparation through training and evaluation.
