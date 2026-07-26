#!/usr/bin/env python3

import subprocess
import sys
from array import array
from pathlib import Path
from typing import Dict, List


FIELD_CASES = {
    "selectionCutFormula": {
        "literal": "1.",
        "pure_tformula": "[global_scale] > 0",
        "hybrid": "([global_scale] > 0) && (OscChannel == 1)",
        "pure_ttreeformula": "(OscChannel == 1) * (RecoEnu > 0.2)",
        "empty": "",
        "omitted": None,
    },
    "nominalTreeWeightFormula": {
        "literal": "1.",
        "pure_tformula": "[global_scale]",
        "hybrid": "([global_scale]) * (OscChannel == 1)",
        "pure_ttreeformula": "(OscChannel == 1) * (RecoEnu > 0.2)",
        "empty": "",
        "omitted": None,
    },
    "sampleWeightFormula": {
        "literal": "1.",
        "pure_tformula": "[global_scale] * [const_norm]",
        "hybrid": "([global_scale]) * (OscChannel == 1)",
        "pure_ttreeformula": "(OscChannel == 1) * (RecoEnu > 0.2)",
        "empty": "",
        "omitted": None,
    },
    "selectionCutStr": {
        "literal": "1.",
        "pure_tformula": "[global_scale] > 0",
        "hybrid": "([global_scale] > 0) && (OscChannel == 1)",
        "pure_ttreeformula": "(OscChannel == 1) * (RecoEnu > 0.2)",
        "empty": "",
        "omitted": None,
    },
    "dialIndexFormula": {
        "literal": "1.",
        "pure_tformula": "[zero_index]",
        "hybrid": "[zero_index] + (OscChannel == 1) - 1",
        "pure_ttreeformula": "(OscChannel == 1) - 1",
        "empty": "",
        "omitted": None,
    },
    "applyCondition": {
        "literal": "1.",
        "pure_tformula": "[global_scale] > 0",
        "hybrid": "([global_scale] > 0) && (OscChannel == 1)",
        "pure_ttreeformula": "(OscChannel == 1) * (RecoEnu > 0.2)",
        "empty": "",
        "omitted": None,
    },
    "variableDictExpr": {
        "literal": "1.",
        "pure_tformula": "[const_norm] * [global_scale]",
        "hybrid": "([global_scale]) * (OscChannel == 1)",
        "pure_ttreeformula": "(OscChannel == 1) * (RecoEnu > 0.2)",
        "empty": "",
        "omitted": None,
    },
}


EXPECTED_RETURN_CODES = {
    (field_name, case_name): 0
    for field_name, case_map in FIELD_CASES.items()
    for case_name in case_map
}
EXPECTED_RETURN_CODES.update(
    {
        ("variableDictExpr", "empty"): 1,
    }
)


def format_return_code(return_code: int) -> str:
    if return_code < 0:
        return f"signal {-return_code}"
    return str(return_code)


def print_summary_table(title: str, rows: List[Dict]) -> None:
    if not rows:
        return

    case_width = max(len("Case"), max(len(row["case"]) for row in rows))
    status_width = len("Status")
    rc_width = max(len("Return"), max(len(format_return_code(row["return_code"])) for row in rows))
    expected_width = max(
        len("Expected"),
        max(len(format_return_code(row["expected_return_code"])) for row in rows),
    )

    print()
    print(title)
    print(
        f"{'Case':<{case_width}}  {'Status':<{status_width}}  "
        f"{'Return':<{rc_width}}  {'Expected':<{expected_width}}"
    )
    print(
        f"{'-' * case_width}  {'-' * status_width}  "
        f"{'-' * rc_width}  {'-' * expected_width}"
    )
    for row in rows:
        print(
            f"{row['case']:<{case_width}}  {row['status']:<{status_width}}  "
            f"{format_return_code(row['return_code']):<{rc_width}}  "
            f"{format_return_code(row['expected_return_code']):<{expected_width}}"
        )


def flush_python_outputs() -> None:
    sys.stdout.flush()
    sys.stderr.flush()


def write_input_root_file(root_path: Path) -> None:
    import uproot

    with uproot.recreate(root_path) as root_file:
        tree = root_file.mktree(
            "tree_mc",
            {
                "X": "float64",
                "OscChannel": "int32",
                "RecoEnu": "float64",
            },
        )
        tree.extend(
            {
                "X": array("d", [-0.5] * 10),
                "OscChannel": array("i", [1] * 10),
                "RecoEnu": array("d", [0.5] * 10),
            }
        )


def build_config_text(field_name: str, case_name: str, root_path: Path) -> str:
    field_value = FIELD_CASES[field_name][case_name]

    variable_dict_lines = [
        '            - { name: "const_norm", expr: "634441." }',
        '            - { name: "global_scale", expr: "2." }',
        '            - { name: "zero_index", expr: "0." }',
    ]

    nominal_weight_formula = "(1.0)"
    selection_cut_formula = "(1)"
    sample_weight_formula = None
    selection_cut_str = None
    dial_index_formula = None
    apply_condition = None

    if field_name == "selectionCutFormula":
        selection_cut_formula = field_value
    elif field_name == "nominalTreeWeightFormula":
        nominal_weight_formula = field_value
    elif field_name == "sampleWeightFormula":
        sample_weight_formula = field_value
    elif field_name == "selectionCutStr":
        selection_cut_str = field_value
    elif field_name == "dialIndexFormula":
        dial_index_formula = field_value
    elif field_name == "applyCondition":
        apply_condition = field_value
    elif field_name == "variableDictExpr":
        if field_value is not None:
            variable_dict_lines.append(
                f'            - {{ name: "resolved_weight", expr: "{field_value}" }}'
            )
            nominal_weight_formula = "[resolved_weight]"
    else:
        raise RuntimeError(f"Unknown field name: {field_name}")

    model_lines = [
        '          tree: tree_mc',
        '          filePathList:',
        f'            - "{root_path}"',
    ]
    if selection_cut_formula is not None:
        model_lines.append(f'          selectionCutFormula: "{selection_cut_formula}"')
    if nominal_weight_formula is not None:
        model_lines.append(f'          nominalTreeWeightFormula: "{nominal_weight_formula}"')
    if dial_index_formula is not None:
        model_lines.append(f'          dialIndexFormula: "{dial_index_formula}"')
    model_block = "\n".join(model_lines)

    sample_extra_lines = []
    if selection_cut_str is not None:
        sample_extra_lines.append(f'            selectionCutStr: "{selection_cut_str}"')
    if sample_weight_formula is not None:
        sample_extra_lines.append(f'            sampleWeightFormula: "{sample_weight_formula}"')

    sample_extra_block = ""
    if sample_extra_lines:
        sample_extra_block = "\n" + "\n".join(sample_extra_lines)

    variable_dict_block = "\n".join(variable_dict_lines)

    parameter_set_block = ""
    if field_name == "applyCondition":
        apply_condition_lines = [
            '                  - dialType: "Normalization"',
            '                    applyOnDataSets: [ "TestSample" ]',
        ]
        if apply_condition is not None:
            apply_condition_lines.append(f'                    applyCondition: "{apply_condition}"')
        apply_condition_block = "\n".join(apply_condition_lines)
        parameter_set_block = f"""
      parametersManagerConfig:
        parameterSetList:
          - name: "NormParameters"
            isEnabled: true
            nominalStepSize: 0.1
            parameterDefinitions:
              - name: "norm_A"
                isEnabled: true
                priorValue: 1.0
                parameterLimits: [0.0, 2.0]
                dialSetDefinitions:
{apply_condition_block}
"""

    return f"""
fitterEngineConfig:
  likelihoodInterfaceConfig:
    jointProbabilityConfig:
      type: PoissonLLH

    dataSetList:
      - name: "TestSample"
        isEnabled: true
        model:
{model_block}
          variableDict:
{variable_dict_block}

    propagatorConfig:
      sampleSetConfig:
        sampleList:
          - name: X
            isEnabled: true
            binning:
              binningDefinition:
                - name: "X"
                  edges: [ -1, 1 ]{sample_extra_block}
            dataSets: [ "TestSample" ]
{parameter_set_block}
"""


def evaluate_case(field_name: str, case_name: str, work_dir: Path, root_path: Path) -> None:
    import GUNDAM

    flush_python_outputs()
    GUNDAM.setRuntimeWorkingDirectory(str(work_dir))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)

    config_builder = GUNDAM.ConfigUtils.ConfigBuilder()
    config_builder.setConfigFromYamlString(build_config_text(field_name, case_name, root_path))
    config_reader = GUNDAM.ConfigUtils.ConfigReader(config_builder.getConfig())
    config_reader.defineField(GUNDAM.ConfigUtils.ConfigReader.FieldDefinition("fitterEngineConfig"))
    fitter_engine_config = config_reader.fetchValueConfigReader("fitterEngineConfig")

    engine = GUNDAM.FitterEngine()
    engine.setConfig(fitter_engine_config)
    engine.configure()
    likelihood_interface = engine.getLikelihoodInterface()
    likelihood_interface.initialize()
    GUNDAM.flushOutput()

    likelihood_interface.propagateAndEvalLikelihood()
    GUNDAM.flushOutput()

    sample = likelihood_interface.getModelPropagator().getSampleSet().getSampleList()[0]
    print(f"{field_name}/{case_name}: accepted")
    print(f"  selectionCutStr = {sample.getSelectionCutsStr()}")
    print(f"  sampleWeightFormula = {sample.getSampleWeightFormulaStr()}")
    print(f"  LLH = {likelihood_interface.getLastLikelihood()}")


def run_case_subprocess(
    field_name: str,
    case_name: str,
    script_path: Path,
    script_dir: Path,
    root_path: Path,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(script_dir),
            "--field",
            field_name,
            "--case",
            case_name,
            "--root-file",
            str(root_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    script_path = Path(__file__).resolve()
    script_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else script_path.parent
    repo_root = script_dir.parents[1]

    if "--field" in sys.argv:
        field_name = sys.argv[sys.argv.index("--field") + 1]
        case_name = sys.argv[sys.argv.index("--case") + 1]
        root_path = Path(sys.argv[sys.argv.index("--root-file") + 1]).resolve()
        evaluate_case(field_name, case_name, Path.cwd(), root_path)
        return 0

    work_dir = Path.cwd()
    root_path = work_dir / "110VariableFormulaResolution.root"
    write_input_root_file(root_path)

    summary_rows = []
    has_failure = False

    for field_name, case_map in FIELD_CASES.items():
        for case_name in case_map:
            expected_return_code = EXPECTED_RETURN_CODES[(field_name, case_name)]
            result = run_case_subprocess(field_name, case_name, script_path, script_dir, root_path)

            print(f"{field_name}/{case_name}: return code = {result.returncode}")
            if result.stdout:
                print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
            if result.stderr:
                print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")

            status = "PASS" if result.returncode == expected_return_code else "FAIL"
            if status == "FAIL":
                has_failure = True

            summary_rows.append(
                {
                    "case": f"{field_name}/{case_name}",
                    "status": status,
                    "return_code": result.returncode,
                    "expected_return_code": expected_return_code,
                }
            )

    passing_rows = [row for row in summary_rows if row["status"] == "PASS"]
    failing_rows = [row for row in summary_rows if row["status"] == "FAIL"]

    print_summary_table("Passing cases", passing_rows)
    print_summary_table("Failing cases", failing_rows)

    if has_failure:
        print("FAIL: formula parsing matrix produced one or more unexpected results.")
        return 1

    print("SUCCESS: formula parsing matrix matches the targeted GUNDAM behavior for all tested fields.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exception:
        print(f"FAIL: {exception}")
        sys.exit(1)
