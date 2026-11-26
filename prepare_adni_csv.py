"""Build the ADNI scan manifest using the metadata workflow in csv_file.ipynb."""

import argparse
from pathlib import Path

import pandas as pd


CLINICAL_COLUMNS = ["RID", "PTID", "EXAMDATE", "AGE", "DX"]
IMAGING_COLUMNS = ["Image Data ID", "Subject", "Group", "Age", "Acq Date"]
MANIFEST_COLUMNS = ["PTID", "DX", "Group", "label", "Image Data ID", "Age", "current_age", "Acq Date"]


def read_clinical_metadata(path):
    """Retain MCI histories and label participants from their recorded diagnoses."""
    clinical = pd.read_csv(path, usecols=CLINICAL_COLUMNS).sort_values("PTID")
    clinical = clinical.loc[clinical["DX"].isin(["MCI", "Dementia"])]
    clinical = clinical.dropna(subset=["PTID"]).copy()

    has_mci = clinical["DX"].eq("MCI").groupby(clinical["PTID"]).transform("any")
    clinical["label"] = clinical.groupby("PTID")["DX"].transform("nunique") - 1
    clinical = clinical.loc[has_mci].dropna().copy()
    clinical["EXAMDATE"] = pd.to_datetime(clinical["EXAMDATE"], format="mixed")
    clinical = clinical.sort_values(["PTID", "EXAMDATE"])

    by_participant = clinical.groupby("PTID")
    elapsed_days = (clinical["EXAMDATE"] - by_participant["EXAMDATE"].transform("first")).dt.days
    baseline_age = by_participant["AGE"].transform("first")
    clinical["current_age"] = (baseline_age + elapsed_days / 365.25).map(lambda age: round(age, 1))
    clinical["EXAM_YEAR_MONTH"] = clinical["EXAMDATE"].dt.to_period("M")
    return clinical


def read_imaging_metadata(path):
    """Normalize the scan export columns used for clinical–imaging matching."""
    scans = pd.read_csv(path, usecols=IMAGING_COLUMNS)
    scans = scans.loc[scans["Group"].ne("CN")].sort_values("Subject").copy()
    scans.loc[scans["Group"].str.contains("MCI", regex=False, na=False), "Group"] = "MCI"
    scans["Acq Date"] = pd.to_datetime(scans["Acq Date"], format="mixed", errors="coerce")
    scans = scans.sort_values(["Subject", "Acq Date"]).rename(columns={"Subject": "PTID"})
    scans["ACQ_YEAR_MONTH"] = scans["Acq Date"].dt.to_period("M")
    return scans


def select_visits(matched_visits):
    """Keep approximately annual visits within four years of the latest match."""
    ordered = matched_visits.sort_values("EXAM_YEAR_MONTH", ascending=False)
    latest_month = ordered["EXAM_YEAR_MONTH"].iloc[0]
    earliest_month = latest_month - pd.DateOffset(years=4)
    previous_month = None
    selected_rows = []
    for index, month in ordered["EXAM_YEAR_MONTH"].items():
        if month < earliest_month:
            break
        if previous_month is None or (previous_month - month).days >= 365:
            selected_rows.append(index)
            previous_month = month
    return ordered.loc[selected_rows]


def fill_missing_years(visits):
    """Fill the five-year grid with copies of the closest later retained visit."""
    visits = visits.copy()
    years = visits["EXAM_YEAR_MONTH"].dt.year
    newest_year = int(years.max())
    visits["slot_year"] = years
    for year in range(newest_year, newest_year - 5, -1):
        if visits["slot_year"].eq(year).any():
            continue
        copied = visits.loc[visits["slot_year"].gt(year)].iloc[-1:].copy()
        copied["EXAM_YEAR_MONTH"] = copied["EXAM_YEAR_MONTH"].map(lambda month: month.replace(year=year))
        copied["slot_year"] = year
        visits = pd.concat([visits, copied], ignore_index=True)
    return visits.sort_values("EXAM_YEAR_MONTH", ascending=False).drop(columns="slot_year")


def build_manifest(adnimerge_path, imaging_path):
    clinical = read_clinical_metadata(adnimerge_path)
    scans = read_imaging_metadata(imaging_path)
    matches = clinical.merge(
        scans, left_on=["PTID", "EXAM_YEAR_MONTH"], right_on=["PTID", "ACQ_YEAR_MONTH"]
    )
    matches["EXAM_YEAR_MONTH"] = matches["EXAM_YEAR_MONTH"].dt.to_timestamp()

    participants = []
    for _, matched_visits in matches.groupby("PTID"):
        visits = select_visits(matched_visits)
        if len(visits) < 2:
            continue
        if visits["label"].iloc[0] == 1 and set(visits["DX"]) != {"MCI", "Dementia"}:
            continue
        participants.append(fill_missing_years(visits)[MANIFEST_COLUMNS])

    if participants:
        return pd.concat(participants, ignore_index=True)
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adnimerge", type=Path, required=True, help="ADNIMERGE clinical CSV")
    parser.add_argument("--imaging-csv", type=Path, required=True, help="ADNI fMRI metadata CSV")
    parser.add_argument("--output", type=Path, required=True, help="destination CSV")
    args = parser.parse_args()
    if args.output.resolve() in {args.adnimerge.resolve(), args.imaging_csv.resolve()}:
        parser.error("--output must be separate from the input files")

    manifest = build_manifest(args.adnimerge, args.imaging_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False, date_format="%Y-%m-%d %H:%M:%S")
    counts = manifest.groupby("label")["PTID"].nunique()
    print(f"Wrote {len(manifest)} rows to {args.output}")
    print(f"Participants: {counts.get(0, 0)} sMCI, {counts.get(1, 0)} pMCI")


if __name__ == "__main__":
    main()
