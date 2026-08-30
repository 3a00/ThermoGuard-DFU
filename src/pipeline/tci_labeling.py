"""TCI Labeling Pipeline — Phase 0.5 of ThermoGuard-DFU.

Computes Thermal Change Index (TCI) scores for all 167 subjects in the
IEEE Plantar Thermogram Database, assigns severity grades to DM subjects,
validates against the published spreadsheet values, and generates the
unsplit subject manifest along with full audit artifacts.

References:
    [TCI-2017] Hernandez-Contreras et al., "A quantitative index for
               classification of plantar thermal changes in the diabetic
               foot," Infrared Physics & Technology, vol. 81, 2017.
    [DB-2019]  Hernandez-Contreras et al., "Plantar Thermogram Database
               for the Study of Diabetic Foot Complications," IEEE Access,
               vol. 7, 2019.
"""

import csv
import glob
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load the project configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Constants (loaded from config at runtime)
# ---------------------------------------------------------------------------

# These are set by main() after loading config.yaml
TCI_REFERENCE = {}
REFERENCE_VERSION = ""
GRADE_THRESHOLDS = []
TEMP_SANITY_MIN = 15.0
TEMP_SANITY_MAX = 45.0
VALIDATION_MAX_MAE = 0.001
VALIDATION_MAX_SINGLE_ERROR = 0.005


# ---------------------------------------------------------------------------
# Step 1 & 2: Angiosome mean extraction
# ---------------------------------------------------------------------------

def angiosome_mean_from_csv(csv_path: str) -> tuple[float, int]:
    """Load a raw angiosome CSV and return the mean temperature and pixel count.

    Background pixels (value 0) and non-finite values are excluded.

    Args:
        csv_path: Absolute path to the angiosome CSV file.

    Returns:
        Tuple of (mean_temperature, valid_pixel_count).

    Raises:
        ValueError: If the CSV contains no valid foot pixels after filtering.
    """
    matrix = np.loadtxt(csv_path, delimiter=',')
    foot_pixels = matrix[np.isfinite(matrix) & (matrix != 0)]

    if foot_pixels.size == 0:
        raise ValueError(f"No valid foot pixels in {csv_path}")

    if np.any(foot_pixels < TEMP_SANITY_MIN) or np.any(foot_pixels > TEMP_SANITY_MAX):
        logging.warning(
            "Temperature outside data-quality bounds (%.1f-%.1f C) in %s: "
            "min=%.2f, max=%.2f",
            TEMP_SANITY_MIN, TEMP_SANITY_MAX, csv_path,
            foot_pixels.min(), foot_pixels.max()
        )

    return float(np.mean(foot_pixels)), int(foot_pixels.size)


# ---------------------------------------------------------------------------
# Step 3: TCI computation
# ---------------------------------------------------------------------------

def compute_tci_foot(angiosome_means: dict[str, float]) -> float:
    """Compute TCI for a single foot from its 4 angiosome mean temperatures.

    Uses the fixed reference vector from [TCI-2017].

    Args:
        angiosome_means: Dict with keys 'LCA', 'LPA', 'MCA', 'MPA'.

    Returns:
        TCI score (float). Higher = greater deviation from healthy baseline.
    """
    total = sum(
        abs(TCI_REFERENCE[region] - angiosome_means[region])
        for region in ['LCA', 'LPA', 'MCA', 'MPA']
    )
    return total / 4.0


def compute_tci_subject(
    left_means: dict[str, float],
    right_means: dict[str, float]
) -> tuple[float, float, float]:
    """Compute subject-level TCI as the mean of left and right foot TCI.

    Args:
        left_means:  Angiosome means for the left foot.
        right_means: Angiosome means for the right foot.

    Returns:
        Tuple of (tci_left, tci_right, tci_subject).
    """
    tci_left = compute_tci_foot(left_means)
    tci_right = compute_tci_foot(right_means)
    tci_subject = (tci_left + tci_right) / 2.0
    return tci_left, tci_right, tci_subject


# ---------------------------------------------------------------------------
# Step 5: Grade assignment
# ---------------------------------------------------------------------------

def assign_severity_grade(
    tci: float,
    group: str,
    thresholds: list[float]
) -> tuple[str, str]:
    """Assign severity grade and model class from TCI score and clinical group.

    CG subjects always receive severity_grade='NA' and model_class='Healthy',
    regardless of TCI. DM subjects are graded 0-4 based on thresholds.

    Args:
        tci:        Subject-level TCI score.
        group:      'CG' (control) or 'DM' (diabetic). Must be exactly one.
        thresholds: Upper boundaries for DM grades 0, 1, 2, 3.

    Returns:
        Tuple of (severity_grade, model_class).

    Raises:
        ValueError: If group is not exactly 'CG' or 'DM'.
    """
    if group == 'CG':
        return 'NA', 'Healthy'

    if group != 'DM':
        raise ValueError(
            f"Unexpected group label: '{group}'. Expected exactly 'CG' or 'DM'. "
            f"Check for typos, extra whitespace, or missing values."
        )

    for grade, upper in enumerate(thresholds):
        if tci <= upper:
            return str(grade), f'DM_Grade{grade}'

    return str(len(thresholds)), f'DM_Grade{len(thresholds)}'


# ---------------------------------------------------------------------------
# Step 6: Grade distribution validation
# ---------------------------------------------------------------------------

def validate_grade_distribution(
    grade_map: dict[str, str],
    min_per_class: int = 3
) -> None:
    """Verify all model_class labels have enough subjects for a 3-way split.

    Args:
        grade_map:     Dict mapping subject_id -> model_class string.
        min_per_class: Minimum subjects per class for splitting.

    Raises:
        ValueError: If any class has fewer than min_per_class subjects.
    """
    counts = Counter(grade_map.values())
    for cls, count in sorted(counts.items()):
        logging.info("  %s: %d subjects", cls, count)
        if count < min_per_class:
            raise ValueError(
                f"Class '{cls}' has only {count} subjects — not enough for "
                f"a 3-way stratified split (need >= {min_per_class}). "
                f"Consider merging with an adjacent grade or adjusting thresholds."
            )


# ---------------------------------------------------------------------------
# Step 1: Build source file inventory
# ---------------------------------------------------------------------------

def build_file_inventory(
    data_root: str
) -> tuple[list[dict], list[dict]]:
    """Scan the extracted data directory and build a file inventory.

    Args:
        data_root: Path to the ThermoDataBase root containing
                   'Control Group' and 'DM Group' subdirectories.

    Returns:
        Tuple of (file_records, subject_records).
        file_records:    List of dicts for file_traceability_manifest.csv.
        subject_records: List of dicts with subject_id, group, gender, folder_path.
    """
    file_records = []
    subject_records = []

    for group_label, group_dir_name in [('CG', 'Control Group'), ('DM', 'DM Group')]:
        group_path = os.path.join(data_root, group_dir_name)
        if not os.path.isdir(group_path):
            logging.error("Group directory not found: %s", group_path)
            continue

        subject_dirs = sorted(glob.glob(os.path.join(group_path, f"{group_label}*")))

        for subj_dir in subject_dirs:
            folder_name = os.path.basename(subj_dir)
            parts = folder_name.split('_')
            subject_id = parts[0]
            gender = parts[1] if len(parts) > 1 else 'Unknown'

            subject_records.append({
                'subject_id': subject_id,
                'group': group_label,
                'gender': gender,
                'folder_path': subj_dir,
                'folder_name': folder_name,
            })

            angio_dir = os.path.join(subj_dir, 'Angiosoms')

            for side in ['L', 'R']:
                for region in ['LCA', 'LPA', 'MCA', 'MPA']:
                    csv_filename = f"{folder_name}_{side}_{region}.csv"
                    csv_path = os.path.join(angio_dir, csv_filename)

                    record = {
                        'subject_id': subject_id,
                        'side': side,
                        'region': region,
                        'csv_path': csv_path,
                        'angiosome_mean': None,
                        'valid_pixel_count': None,
                        'status': 'valid',
                        'status_detail': '',
                    }

                    if not os.path.isfile(csv_path):
                        record['status'] = 'missing_csv'
                        record['status_detail'] = f"Missing: {csv_filename}"
                        logging.warning("Missing CSV: %s", csv_path)
                    else:
                        try:
                            mean_temp, pixel_count = angiosome_mean_from_csv(csv_path)
                            record['angiosome_mean'] = round(mean_temp, 10)
                            record['valid_pixel_count'] = pixel_count
                        except ValueError as e:
                            record['status'] = 'zero_foot_pixels'
                            record['status_detail'] = str(e)
                            logging.warning("Zero foot pixels: %s", e)

                    file_records.append(record)

    return file_records, subject_records


# ---------------------------------------------------------------------------
# Step 4: Validate against spreadsheet
# ---------------------------------------------------------------------------

def load_spreadsheet_tci(excel_path: str) -> dict[str, dict[str, float]]:
    """Load per-foot TCI values from the Excel spreadsheet.

    Args:
        excel_path: Path to 'Plantar Thermogram Database.xlsx'.

    Returns:
        Dict mapping (subject_id, side) -> spreadsheet_tci.
        Key format: "{subject_id}_{side}" e.g. "DM001_R".
    """
    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    tci_map = {}

    for sheet_name in ['Control Group', 'DM Group']:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))

        for row in rows[2:]:
            if row[0] is None:
                continue
            subject_id = str(row[0]).strip()

            tci_right = row[11]
            tci_left = row[17]

            if isinstance(tci_right, (int, float)):
                tci_map[f"{subject_id}_R"] = float(tci_right)
            if isinstance(tci_left, (int, float)):
                tci_map[f"{subject_id}_L"] = float(tci_left)

    wb.close()
    return tci_map


def validate_tci(
    file_records: list[dict],
    spreadsheet_tci: dict[str, float],
    output_dir: str,
    config: dict,
) -> tuple[list[dict], dict]:
    """Validate computed TCI against spreadsheet for every foot.

    Args:
        file_records:     Output of build_file_inventory().
        spreadsheet_tci:  Output of load_spreadsheet_tci().
        output_dir:       Directory to write validation reports.
        config:           Full config dict.

    Returns:
        Tuple of (validation_rows, summary_dict).
    """
    # Group file records by (subject_id, side)
    foot_data = {}
    for rec in file_records:
        key = f"{rec['subject_id']}_{rec['side']}"
        if key not in foot_data:
            foot_data[key] = {}
        foot_data[key][rec['region']] = rec

    validation_rows = []
    errors = []
    mismatches = []

    for key, regions in sorted(foot_data.items()):
        subject_id, side = key.rsplit('_', 1)

        # Check all 4 regions have valid means
        all_valid = all(
            regions.get(r, {}).get('angiosome_mean') is not None
            for r in ['LCA', 'LPA', 'MCA', 'MPA']
        )

        if not all_valid:
            validation_rows.append({
                'subject_id': subject_id,
                'foot_side': side,
                'spreadsheet_tci': spreadsheet_tci.get(key, ''),
                'computed_tci': '',
                'absolute_error': '',
                'status': 'incomplete_data',
            })
            continue

        means = {r: regions[r]['angiosome_mean'] for r in ['LCA', 'LPA', 'MCA', 'MPA']}
        computed = compute_tci_foot(means)
        sheet_val = spreadsheet_tci.get(key)

        if sheet_val is None:
            validation_rows.append({
                'subject_id': subject_id,
                'foot_side': side,
                'spreadsheet_tci': '',
                'computed_tci': f"{computed:.10f}",
                'absolute_error': '',
                'status': 'no_spreadsheet_value',
            })
            continue

        error = abs(computed - sheet_val)
        errors.append(error)

        status = 'valid'
        if error > VALIDATION_MAX_SINGLE_ERROR:
            status = 'mismatch'
            mismatches.append({
                'subject_id': subject_id,
                'foot_side': side,
                'spreadsheet_tci': f"{sheet_val:.10f}",
                'computed_tci': f"{computed:.10f}",
                'absolute_error': f"{error:.10f}",
            })

        validation_rows.append({
            'subject_id': subject_id,
            'foot_side': side,
            'spreadsheet_tci': f"{sheet_val:.10f}",
            'computed_tci': f"{computed:.10f}",
            'absolute_error': f"{error:.10f}",
            'status': status,
        })

    # Build summary
    err_arr = np.array(errors) if errors else np.array([0.0])
    mae = float(err_arr.mean())
    max_err = float(err_arr.max())
    passed = (mae < VALIDATION_MAX_MAE) and (max_err < VALIDATION_MAX_SINGLE_ERROR)

    summary = {
        'total_feet': len(foot_data),
        'valid_comparisons': len(errors),
        'incomplete_data': sum(1 for r in validation_rows if r['status'] == 'incomplete_data'),
        'no_spreadsheet_value': sum(1 for r in validation_rows if r['status'] == 'no_spreadsheet_value'),
        'mean_absolute_error': mae,
        'max_absolute_error': max_err,
        'mismatches_above_threshold': len(mismatches),
        'threshold': VALIDATION_MAX_SINGLE_ERROR,
        'pass': passed,
        'reference_version': REFERENCE_VERSION,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    # Write validation report
    report_path = os.path.join(output_dir, 'tci_validation_report.csv')
    with open(report_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'subject_id', 'foot_side', 'spreadsheet_tci',
            'computed_tci', 'absolute_error', 'status'
        ])
        writer.writeheader()
        writer.writerows(validation_rows)
    logging.info("Wrote validation report: %s (%d rows)", report_path, len(validation_rows))

    # Write summary JSON
    summary_path = os.path.join(output_dir, 'tci_validation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info("Wrote validation summary: %s", summary_path)

    # Write mismatch report
    mismatch_path = os.path.join(output_dir, 'tci_mismatch_report.csv')
    with open(mismatch_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'subject_id', 'foot_side', 'spreadsheet_tci',
            'computed_tci', 'absolute_error'
        ])
        writer.writeheader()
        writer.writerows(mismatches)
    logging.info("Wrote mismatch report: %s (%d rows)", mismatch_path, len(mismatches))

    return validation_rows, summary


# ---------------------------------------------------------------------------
# Step 7: Distribution plot
# ---------------------------------------------------------------------------

def generate_distribution_plot(
    manifest_rows: list[dict],
    thresholds: list[float],
    output_path: str,
) -> None:
    """Generate a TCI distribution histogram with grade threshold lines.

    Args:
        manifest_rows: List of manifest dicts with 'group' and 'tci_subject'.
        thresholds:    Grade boundary values for vertical lines.
        output_path:   Path to save the PNG file.
    """
    cg_tci = [float(r['tci_subject']) for r in manifest_rows if r['group'] == 'CG']
    dm_tci = [float(r['tci_subject']) for r in manifest_rows if r['group'] == 'DM']

    fig, ax = plt.subplots(figsize=(12, 6))

    bins = np.linspace(0, 10, 41)
    ax.hist(cg_tci, bins=bins, alpha=0.6, label=f'Control (CG, n={len(cg_tci)})',
            color='#2196F3', edgecolor='white')
    ax.hist(dm_tci, bins=bins, alpha=0.6, label=f'Diabetic (DM, n={len(dm_tci)})',
            color='#F44336', edgecolor='white')

    grade_labels = ['G0', 'G1', 'G2', 'G3', 'G4']
    colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336', '#9C27B0']
    for i, t in enumerate(thresholds):
        ax.axvline(x=t, color=colors[i], linestyle='--', linewidth=1.5,
                   label=f'Grade {i}/{i+1} boundary (TCI={t})')

    ax.set_xlabel('TCI Score (Subject-Level)', fontsize=12)
    ax.set_ylabel('Number of Subjects', fontsize=12)
    ax.set_title('TCI Distribution by Group — IEEE Plantar Thermogram Database', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info("Wrote distribution plot: %s", output_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """Execute the full Phase 0.5 TCI labeling pipeline."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Determine project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config.yaml')
    logging.info("Project root: %s", project_root)

    # Load config
    config = load_config(config_path)
    logging.info("Loaded config from: %s", config_path)

    # Set globals from config
    global TCI_REFERENCE, REFERENCE_VERSION, GRADE_THRESHOLDS
    global TEMP_SANITY_MIN, TEMP_SANITY_MAX
    global VALIDATION_MAX_MAE, VALIDATION_MAX_SINGLE_ERROR

    TCI_REFERENCE = config['tci']['reference_vector']
    REFERENCE_VERSION = config['tci']['reference_version']
    GRADE_THRESHOLDS = config['tci']['grade_thresholds']
    TEMP_SANITY_MIN = config['tci']['sanity_bounds']['min']
    TEMP_SANITY_MAX = config['tci']['sanity_bounds']['max']
    VALIDATION_MAX_MAE = config['tci']['validation']['max_mae']
    VALIDATION_MAX_SINGLE_ERROR = config['tci']['validation']['max_single_error']

    data_root = config['data']['extracted_root']
    excel_path = config['data']['excel_path']
    output_dir = os.path.join(project_root, 'outputs')
    data_dir = os.path.join(project_root, 'data')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1 & 2: Build file inventory and compute angiosome means
    # -----------------------------------------------------------------------
    logging.info("=" * 60)
    logging.info("STEP 1-2: Building file inventory and computing angiosome means")
    logging.info("=" * 60)

    file_records, subject_records = build_file_inventory(data_root)

    # Write file traceability manifest
    file_manifest_path = os.path.join(data_dir, 'file_traceability_manifest.csv')
    with open(file_manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'subject_id', 'side', 'region', 'csv_path',
            'angiosome_mean', 'valid_pixel_count', 'status', 'status_detail'
        ])
        writer.writeheader()
        writer.writerows(file_records)
    logging.info("Wrote file traceability manifest: %s (%d records)",
                 file_manifest_path, len(file_records))

    logging.info("Found %d subjects (%d CG, %d DM)",
                 len(subject_records),
                 sum(1 for s in subject_records if s['group'] == 'CG'),
                 sum(1 for s in subject_records if s['group'] == 'DM'))

    # -----------------------------------------------------------------------
    # Step 3: Compute TCI per subject
    # -----------------------------------------------------------------------
    logging.info("=" * 60)
    logging.info("STEP 3: Computing TCI scores")
    logging.info("=" * 60)

    # Group file records by subject and side for TCI computation
    subj_angio = {}
    for rec in file_records:
        key = (rec['subject_id'], rec['side'])
        if key not in subj_angio:
            subj_angio[key] = {}
        if rec['angiosome_mean'] is not None:
            subj_angio[key][rec['region']] = rec['angiosome_mean']

    manifest_rows = []
    model_class_map = {}

    for subj_info in subject_records:
        sid = subj_info['subject_id']
        group = subj_info['group']
        gender = subj_info['gender']

        left_key = (sid, 'L')
        right_key = (sid, 'R')

        left_means = subj_angio.get(left_key, {})
        right_means = subj_angio.get(right_key, {})

        # Determine calculation status
        calc_status = 'valid'
        status_detail = ''

        missing_regions = []
        for side, means in [('L', left_means), ('R', right_means)]:
            for region in ['LCA', 'LPA', 'MCA', 'MPA']:
                if region not in means:
                    missing_regions.append(f"{side}_{region}")

        if missing_regions:
            calc_status = 'missing_csv'
            status_detail = f"Missing regions: {', '.join(missing_regions)}"
            logging.warning("Subject %s: %s", sid, status_detail)

        # Compute TCI if all regions available
        if len(left_means) == 4 and len(right_means) == 4:
            tci_left, tci_right, tci_subject = compute_tci_subject(left_means, right_means)
        elif len(left_means) == 4:
            tci_left = compute_tci_foot(left_means)
            tci_right = ''
            tci_subject = tci_left
            calc_status = 'warning_incomplete'
            status_detail = 'Only left foot available'
        elif len(right_means) == 4:
            tci_right = compute_tci_foot(right_means)
            tci_left = ''
            tci_subject = tci_right
            calc_status = 'warning_incomplete'
            status_detail = 'Only right foot available'
        else:
            tci_left = ''
            tci_right = ''
            tci_subject = ''
            calc_status = 'missing_csv'
            status_detail = 'Insufficient data for TCI computation'
            logging.error("Subject %s: Cannot compute TCI — %s", sid, status_detail)

        # Assign grade
        if tci_subject != '':
            severity_grade, model_class = assign_severity_grade(
                float(tci_subject), group, GRADE_THRESHOLDS
            )
        else:
            severity_grade = 'NA'
            model_class = 'Unknown'

        model_class_map[sid] = model_class

        # Compute validation difference
        val_diff = ''
        if tci_subject != '':
            # Will be refined in Step 4 — placeholder for now
            val_diff = 0.0

        manifest_rows.append({
            'subject_id': sid,
            'group': group,
            'gender': gender,
            'tci_right': f"{tci_right:.10f}" if isinstance(tci_right, float) else str(tci_right),
            'tci_left': f"{tci_left:.10f}" if isinstance(tci_left, float) else str(tci_left),
            'tci_subject': f"{tci_subject:.10f}" if isinstance(tci_subject, float) else str(tci_subject),
            'severity_grade': severity_grade,
            'model_class': model_class,
            'reference_version': REFERENCE_VERSION,
            'calculation_status': calc_status,
            'status_detail': status_detail,
            'validation_difference': '',
            'split': '',
        })

    logging.info("Computed TCI for %d subjects", len(manifest_rows))

    # -----------------------------------------------------------------------
    # Step 4: Validate against spreadsheet
    # -----------------------------------------------------------------------
    logging.info("=" * 60)
    logging.info("STEP 4: Validating against spreadsheet")
    logging.info("=" * 60)

    spreadsheet_tci = load_spreadsheet_tci(excel_path)
    logging.info("Loaded %d TCI values from spreadsheet", len(spreadsheet_tci))

    validation_rows, summary = validate_tci(
        file_records, spreadsheet_tci, output_dir, config
    )

    # Update manifest with validation differences
    foot_errors = {}
    for vr in validation_rows:
        key = vr['subject_id']
        if vr['absolute_error'] and vr['status'] != 'incomplete_data':
            if key not in foot_errors:
                foot_errors[key] = []
            foot_errors[key].append(float(vr['absolute_error']))

    for row in manifest_rows:
        sid = row['subject_id']
        if sid in foot_errors:
            avg_err = np.mean(foot_errors[sid])
            row['validation_difference'] = f"{avg_err:.10f}"
            if avg_err > VALIDATION_MAX_SINGLE_ERROR:
                row['calculation_status'] = 'excel_mismatch'
                row['status_detail'] = f"Validation error: {avg_err:.6f}"

    if summary['pass']:
        logging.info("VALIDATION PASSED: MAE=%.10f, Max=%.10f, Mismatches=%d",
                      summary['mean_absolute_error'],
                      summary['max_absolute_error'],
                      summary['mismatches_above_threshold'])
    else:
        logging.error("VALIDATION FAILED: MAE=%.10f, Max=%.10f, Mismatches=%d",
                       summary['mean_absolute_error'],
                       summary['max_absolute_error'],
                       summary['mismatches_above_threshold'])

    # -----------------------------------------------------------------------
    # Step 5 & 6: Grade assignment validation
    # -----------------------------------------------------------------------
    logging.info("=" * 60)
    logging.info("STEP 5-6: Validating grade distribution")
    logging.info("=" * 60)

    valid_classes = {k: v for k, v in model_class_map.items() if v != 'Unknown'}
    validate_grade_distribution(valid_classes)

    # -----------------------------------------------------------------------
    # Step 7: Distribution plot
    # -----------------------------------------------------------------------
    logging.info("=" * 60)
    logging.info("STEP 7: Generating distribution plot")
    logging.info("=" * 60)

    plot_path = os.path.join(output_dir, 'tci_distribution_by_group.png')
    valid_manifest = [r for r in manifest_rows if r['tci_subject'] != '']
    generate_distribution_plot(valid_manifest, GRADE_THRESHOLDS, plot_path)

    # -----------------------------------------------------------------------
    # Step 8: Save main manifest
    # -----------------------------------------------------------------------
    logging.info("=" * 60)
    logging.info("STEP 8: Saving subject manifest (unsplit)")
    logging.info("=" * 60)

    manifest_path = os.path.join(data_dir, 'subject_manifest_unsplit.csv')
    fieldnames = [
        'subject_id', 'group', 'gender',
        'tci_right', 'tci_left', 'tci_subject',
        'severity_grade', 'model_class',
        'reference_version', 'calculation_status', 'status_detail',
        'validation_difference', 'split'
    ]
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    logging.info("Wrote subject manifest: %s (%d rows)", manifest_path, len(manifest_rows))

    # -----------------------------------------------------------------------
    # Step 9: Summary report
    # -----------------------------------------------------------------------
    logging.info("=" * 60)
    logging.info("PHASE 0.5 SUMMARY REPORT")
    logging.info("=" * 60)

    logging.info("Total subjects processed: %d", len(manifest_rows))

    class_counts = Counter(r['model_class'] for r in manifest_rows)
    logging.info("Class distribution:")
    for cls, count in sorted(class_counts.items()):
        logging.info("  %-15s %d subjects", cls, count)

    status_counts = Counter(r['calculation_status'] for r in manifest_rows)
    if any(s != 'valid' for s in status_counts):
        logging.info("Calculation status flags:")
        for status, count in sorted(status_counts.items()):
            if status != 'valid':
                logging.warning("  %-25s %d subjects", status, count)
    else:
        logging.info("All subjects: calculation_status = valid")

    logging.info("Validation: MAE=%.10f, Max Error=%.10f",
                 summary['mean_absolute_error'], summary['max_absolute_error'])
    logging.info("Overall status: %s", "PASS" if summary['pass'] else "FAIL")

    logging.info("=" * 60)
    logging.info("Output files:")
    logging.info("  %s", file_manifest_path)
    logging.info("  %s", manifest_path)
    logging.info("  %s", os.path.join(output_dir, 'tci_validation_report.csv'))
    logging.info("  %s", os.path.join(output_dir, 'tci_validation_summary.json'))
    logging.info("  %s", os.path.join(output_dir, 'tci_mismatch_report.csv'))
    logging.info("  %s", plot_path)
    logging.info("=" * 60)

    return 0 if summary['pass'] else 1


if __name__ == "__main__":
    sys.exit(main())
