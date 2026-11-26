# Model configurations

[5DfCF](5DfCF/) contains the Tiny, Small, Base, and Large YAML presets. [5DfCF_pp](5DfCF_pp/) contains the alternate presets with separate attention grouping settings.

Each file stores architecture settings under `MODEL`. The `MODEL.CROS` block gives the stage depths, attention head counts, patch sizes, and grouping settings. Record the exact preset used for a checkpoint so its architecture can be recovered. See the [model source overview](../models/README.md) for the corresponding code.
