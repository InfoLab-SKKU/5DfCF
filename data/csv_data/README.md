# ADNI CSV metadata

The CSV preprocessing script reads `ADNIMERGE_12Dec2024.csv` for clinical visits and `chris_5D_fmri_adni_final_12_31_2024.csv` for scan metadata. The [repository instructions](../../README.md#1-prepare-the-adni-csv-manifest) give the command for rebuilding the manifest.

[adni_csv_final](adni_csv_final/) contains the derived manifests. The original notebook's intermediate exports are under [adni_csv_fmri_processed](adni_csv_fmri_processed/) and [new_adni_processed](new_adni_processed/).

Each output row represents one annual slot. Missing slots may reuse a scan and its acquisition date, so count unique `PTID` values for participant totals and unique image IDs for scan totals.
