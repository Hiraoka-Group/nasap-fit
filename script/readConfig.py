#!/usr/bin/env python3
import os
import sys
import yaml
import csv

# filepath: /home/aspi/Project/nasap-fit-cpp/script/readConfig.py

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "M9L6", "config.yaml"))
DEFAULT_OUTPUT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "include", "constants.hpp"))

#double型の出力
def format_double(x):
    return format(float(x), '.17g')

#bool型の受け取りと検証
def require_bool(name, x):
    if x is None:
        raise ValueError(f"{name} is required")
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ('1', 'true', 'yes', 'y'):
        return True
    if s in ('0', 'false', 'no', 'n'):
        return False
    raise ValueError(f"{name} must be one of 1,true,yes,y or 0,false,no,n (got {x!r})")

#整数値の取得と検証
def require_int(name, x, min_value=0):
    if x is None:
        raise ValueError(f"{name} is required")
    try:
        v = int(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer (got {x!r})")
    if min_value is not None and v < min_value:
        raise ValueError(f"{name} must be >= {min_value} (got {v})")
    return v

#浮動小数点値の取得と検証
def require_float(name, x, min_value=None, max_value=None):
    if x is None:
        raise ValueError(f"{name} is required")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number (got {x!r})")
    if min_value is not None and v < min_value:
        raise ValueError(f"{name} must be >= {min_value} (got {v})")
    if max_value is not None and v > max_value:
        raise ValueError(f"{name} must be <= {max_value} (got {v})")
    return v

#浮動小数点値の取得と検証（任意の場合）
def optional_float(name, x, default=None, min_value=None, max_value=None):
    if x is None:
        if default is None:
            return None
        return float(default)
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number (got {x!r})")
    if min_value is not None and v < min_value:
        raise ValueError(f"{name} must be >= {min_value} (got {v})")
    if max_value is not None and v > max_value:
        raise ValueError(f"{name} must be <= {max_value} (got {v})")
    return v

# helper to check if string represents an integer
def _is_int_string(s):
    s2 = str(s).strip()
    if s2 == '':
        return False
    try:
        f = float(s2)
    except ValueError:
        return False
    return f.is_integer()

# validate QASAP data CSV file
def validate_qasap(path, tracked_names):
    errors = []
    if not os.path.isfile(path):
        errors.append(f"QASAPDataFile not found: {path}")
        raise ValueError("\n".join(errors))

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"QASAPDataFile has no header: {path}")
        fields = [fn.strip() for fn in reader.fieldnames]
        required_cols = set(tracked_names)
        missing_cols = required_cols - set(fields)
        if missing_cols:
            errors.append(f"QASAPDataFile missing required columns: {sorted(missing_cols)}")
        if reader.fieldnames[0] in required_cols:
            errors.append("first column must represent time, not a tracked species")

        # check tracked species present in header
        missing = [n for n in tracked_names if n not in fields]
        if missing:
            errors.append(f"Tracked species missing from QASAP header: {missing}")

        # Validate data rows: monotonic increasing time, all values numeric and present
        prev_time = None
        row_no = 1  # header is row 1
        for row in reader:
            row_no += 1
            # allow rows with trailing empty cells trimmed by CSV reader? check length match header
            for col_key in required_cols:
                try:
                    cell = row[col_key]
                except IndexError:
                    errors.append(f"Row {row_no} Col {col_key} ('{row[col_key]}'): missing")
                    break
                if str(cell).strip() == '':
                    errors.append(f"Row {row_no} Col {col_key} ('{row[col_key]}'): missing")
                    break
                try:
                    val = float(cell)
                except ValueError:
                    errors.append(f"Row {row_no} Col {col_key} ('{row[col_key]}'): not numeric ('{cell}')")
                    break
                if col_key == 'time':
                    # time monotonic increasing (strict)
                    if prev_time is not None and val <= prev_time:
                        errors.append(f"Row {row_no} time {val} is not strictly greater than previous time {prev_time}")
                    prev_time = val

    if errors:
        raise ValueError("QASAPDataFile validation failed:\n" + "\n".join(errors))

# validate reaction network CSV file
def validate_reaction(path, species,constant_size):
    required_cols = {'init_assem_id', 'entering_assem_id', 'product_assem_id', 'leaving_assem_id', 'duplicate_count', 'kind'}
    errors = []
    if not os.path.isfile(path):
        errors.append(f"reactionDataFile not found: {path}")
        raise ValueError("\n".join(errors))

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"reactionDataFile has no header: {path}")
        fields = [fn.strip() for fn in reader.fieldnames]
        missing_cols = required_cols - set(fields)
        if missing_cols:
            errors.append(f"reactionDataFile missing required columns: {sorted(missing_cols)}")

        # validate rows]
        reactKinds = set()
        row_no = 1  # header row
        for row in reader:
            row_no += 1
            # init and product must not be missing
            for col in ('init_assem_id', 'product_assem_id'):
                val = row.get(col, "")
                if val is None or str(val).strip() == '':
                    errors.append(f"Row {row_no}: column '{col}' is missing")
            # check integer range for the four id columns if present
            for col in ('init_assem_id', 'entering_assem_id', 'product_assem_id', 'leaving_assem_id'):
                raw = row.get(col, "")
                s = str(raw).strip()
                if s == '':
                    continue  # missing is allowed for entering/leaving
                if not _is_int_string(s):
                    errors.append(f"Row {row_no} column '{col}': not an integer ('{raw}')")
                    break
                try:
                    v = int(float(s))
                except Exception:
                    errors.append(f"Row {row_no} column '{col}': cannot parse integer ('{raw}')")
                    break
                if v < 0 or v >= species:
                    errors.append(f"Row {row_no} column '{col}': value {v} out of range [0, {species-1}]")
            # kind must be present
            kind_raw = row.get('kind', "")
            kind = str(kind_raw).strip()
            if kind == '':
                errors.append(f"Row {row_no}: column 'kind' is missing")
            else:
                reactKinds.add(kind)

            # duplicate_count must be integer >= 1
            dup_raw = row.get('duplicate_count', "")
            dup_s = str(dup_raw).strip()
            if dup_s == '':
                errors.append(f"Row {row_no}: column 'duplicate_count' is missing")
            elif not _is_int_string(dup_s):
                errors.append(f"Row {row_no} column 'duplicate_count': not an integer ('{dup_raw}')")
            else:
                try:
                    dup_v = int(float(dup_s))
                except Exception:
                    errors.append(f"Row {row_no} column 'duplicate_count': cannot parse integer ('{dup_raw}')")
                else:
                    if dup_v < 1:
                        errors.append(f"Row {row_no} column 'duplicate_count': value {dup_v} must be >= 1")
        # check that number of unique reaction kinds matches constant_size
        if len(reactKinds) != constant_size:
            errors.append(f"reactionDataFile: number of unique reaction kinds {len(reactKinds)} does not match constantSize {constant_size}; kinds found: {sorted(reactKinds)}")


    if errors:
        raise ValueError("reactionDataFile validation failed:\n" + "\n".join(errors))


def main(cfg_path=DEFAULT_INPUT, out_path=DEFAULT_OUTPUT):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    QASAP_path = cfg.get('QASAPDataFile', '')
    if QASAP_path == None or QASAP_path.strip() == '':
        raise ValueError("QASAPDataFile is required and cannot be empty")
    reaction_path = cfg.get('reactionDataFile', '')
    if reaction_path == None or reaction_path.strip() == '':
        raise ValueError("reactionDataFile is required and cannot be empty")

    # Required integer params
    species = require_int('species', cfg.get('species'), min_value=1)
    constant_size = require_int('constantSize', cfg.get('constantSize'), min_value=1)
    pop_size = require_int('popSize', cfg.get('popSize'), min_value=1)
    max_gen = require_int('maxGen', cfg.get('maxGen'), min_value=1)

    # Optional numeric params with validation and defaults
    tol_abs = optional_float('tolAbsError', cfg.get('tolAbsError'), default=1e-10, min_value=0.0, max_value=1.0)
    tol_rel = optional_float('tolRelError', cfg.get('tolRelError'), default=1e-6, min_value=0.0, max_value=1.0)
    # scalar and crossOver default to 0.5 if not provided; must be in [0,1]
    scalar = optional_float('scalar', cfg.get('scalar'), default=0.5, min_value=0.0, max_value=1.0)
    cross = optional_float('crossOver', cfg.get('crossOver'), default=0.5, min_value=0.0, max_value=1.0)
    upper = optional_float('upperLim', cfg.get('upperLim'), default=1e4, min_value=0.0)
    lower = optional_float('lowerLim', cfg.get('lowerLim'), default=1e-3, min_value=0.0)

    usePreGeneratedRhsf = require_bool('usePreGeneratedRhsf', cfg.get('usePreGeneratedRhsf', True))
    usePreGeneratedJacobian = require_bool('usePreGeneratedJacobian', cfg.get('usePreGeneratedJacobian', True))
    

    tracked = cfg.get('trackedSpecies') or {}
    # Preserve YAML order for tracked species
    tracked_items = list(tracked.items())
    tracked_count = len(tracked_items)
    tracked_names = [name for name, _ in tracked_items]
    tracked_indices = []
    full_concs = []
    for name, info in tracked_items:
        if not isinstance(info, dict):
            raise ValueError(f"tracked species '{name}' info must be a mapping")
        idx = info.get('index')
        conc_key = "100%concentration"
        conc_val = info.get(conc_key)
        if idx is None:
            raise ValueError(f"tracked species '{name}' missing index")
        idx_i = require_int(f"trackedSpecies.{name}.index", idx, min_value=0)
        if idx_i >= species:
            raise ValueError(f"tracked species '{name}' index {idx_i} out of range (species={species})")
        if conc_val is None:
            raise ValueError(f"tracked species '{name}' missing 100% concentration")
        conc_f = optional_float(f"trackedSpecies.{name}.{conc_key}", conc_val, default=0.0, min_value=0.0)
        tracked_indices.append(idx_i)
        full_concs.append(format_double(conc_f))

    # Build C++ content
    lines = []
    lines.append('#pragma once')
    lines.append('#include <string>')
    lines.append('#include <string_view>')
    lines.append('')
    lines.append(f'#define USE_PREGENERATED_RHSF ({1 if usePreGeneratedRhsf else 0})')
    lines.append(f'#define USE_PREGENERATED_JACOBIAN ({1 if usePreGeneratedJacobian else 0})')
    lines.append('')
    lines.append('namespace config {')
    lines.append(f'const std::string QASAPFile = "../{QASAP_path}";')
    lines.append(f'const std::string reactNetworkFile = "../{reaction_path}";')
    lines.append('')
    lines.append('//化学種の数')
    lines.append(f'constexpr int species = {species};')
    lines.append('//反応速度定数の数')
    lines.append(f'constexpr int constantSize = {constant_size};')
    lines.append('')
    lines.append(f'//データにより与えられる化学種の数')
    lines.append(f'const int trackedSpecies={tracked_count};')
    names_list = ', '.join(f'"{name}"' for name in tracked_names)
    lines.append(f'//csvファイル内の化学種の名前')
    lines.append(f'const std::string_view trackedNames[] = {{ {names_list} }};')
    idx_list = ', '.join(str(i) for i in tracked_indices)
    lines.append(f'//データにより与えられる化学種のindex')
    lines.append(f'const int trackedIndex[] = {{ {idx_list} }};')
    conc_list = ', '.join(full_concs)
    lines.append(f'//Table_S1.csvにおいて、化学種の100%にあたる濃度')
    lines.append(f'const double fullConc[] = {{{conc_list}}};')
    lines.append('')
    lines.append('')
    lines.append('//差分進化法のエージェント数')
    lines.append(f'const int popSize = {pop_size};')
    lines.append('//差分進化法の最大世代数')
    lines.append(f'const int maxGen = {max_gen};')
    lines.append('')
    lines.append('//シミュレーションの許容絶対誤差')
    lines.append(f'const double tolAbsError = {format_double(tol_abs)};')
    lines.append('//シミュレーションの許容相対誤差')
    lines.append(f'const double tolRelError = {format_double(tol_rel)};')
    lines.append('//適応型ルンゲクッタ法の安全係数')
    lines.append('const double safetyConstant=0.9;')
    lines.append('//差分進化法のパラメータ')
    lines.append(f'const double scalar = {format_double(scalar)}, crossOver = {format_double(cross)};')

    lines.append('//反応速度定数の上限下限')
    lines.append(f'const double upperLim = {format_double(upper)}, lowerLim = {format_double(lower)};')

    lines.append('}')
    content = '\n'.join(lines) + '\n'

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Wrote {out_path}")

    # Additional validation of input files
    try:
        # resolve relative paths relative to config file directory
        rootFolder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        QASAP_path = QASAP_path if os.path.isabs(str(QASAP_path)) else os.path.normpath(os.path.join(rootFolder_path, str(QASAP_path)))
        reaction_path = reaction_path if os.path.isabs(str(reaction_path)) else os.path.normpath(os.path.join(rootFolder_path, str(reaction_path)))

        validate_qasap(QASAP_path, tracked_names)
        validate_reaction(reaction_path, species, constant_size)
        print("Input files validation: OK")
    except Exception as e:
        print("Validation error:", file=sys.stderr)
        print(str(e), file=sys.stderr)
        sys.exit(1)

    

if __name__ == '__main__':
    cfg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    main(cfg, out)

    
