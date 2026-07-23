#!/usr/bin/env python3

import subprocess
import sys
from array import array
from pathlib import Path


FIELD_CASES = {
    "selectionCutFormula": {
        "pure_tformula": "[global_scale] > 0",
        "hybrid": "([global_scale] > 0) && (OscChannel == 1)",
        "pure_ttreeformula": "(OscChannel == 1) * (RecoEnu > 0.2)",
    },
    "nominalTreeWeightFormula": {
        "pure_tformula": "[global_scale]",
        "hybrid": "([global_scale]) * (OscChannel == 1)",
        "pure_ttreeformula": "(OscChannel == 1) * (RecoEnu > 0.2)",
    },
    "sampleWeightFormula": {
        "pure_tformula": "[global_scale] * [const_norm]",
        "hybrid": "([global_scale]) * (OscChannel == 1)",
        "pure_ttreeformula": "(OscChannel == 1) * (RecoEnu > 0.2)",
    },
    "selectionCutStr": {
        "pure_tformula": "[global_scale] > 0",
        "hybrid": "([global_scale] > 0) && (OscChannel == 1)",
        "pure_ttreeformula": "(OscChannel == 1) * (RecoEnu > 0.2)",
    },
    "variableDictExpr": {
        "pure_tformula": "[const_norm] * [global_scale]",
        "hybrid": "([global_scale]) * (OscChannel == 1)",
        "pure_ttreeformula": "(OscChannel == 1) * (RecoEnu > 0.2)",
    },
}


EXPECTED_RETURN_CODES = {
    (field_name, case_name): 0
    for field_name, case_map in FIELD_CASES.items()
    for case_name in case_map
}


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
    variable_dict_lines = [
        '            - { name: "const_norm", expr: "634441." }',
        '            - { name: "global_scale", expr: "2." }',
    ]

    nominal_weight_formula = "(1.0)"
    selection_cut_formula = "(1)"
    sample_weight_formula = ""
    selection_cut_str = ""

    if field_name == "selectionCutFormula":
        selection_cut_formula = FIELD_CASES[field_name][case_name]
    elif field_name == "nominalTreeWeightFormula":
        nominal_weight_formula = FIELD_CASES[field_name][case_name]
    elif field_name == "sampleWeightFormula":
        sample_weight_formula = FIELD_CASES[field_name][case_name]
    elif field_name == "selectionCutStr":
        selection_cut_str = FIELD_CASES[field_name][case_name]
    elif field_name == "variableDictExpr":
        variable_dict_lines.append(
            f'            - {{ name: "resolved_weight", expr: "{FIELD_CASES[field_name][case_name]}" }}'
        )
        nominal_weight_formula = "resolved_weight"
    else:
        raise RuntimeError(f"Unknown field name: {field_name}")

    sample_extra_lines = []
    if selection_cut_str:
        sample_extra_lines.append(f'            selectionCutStr: "{selection_cut_str}"')
    if sample_weight_formula:
        sample_extra_lines.append(f'            sampleWeightFormula: "{sample_weight_formula}"')

    sample_extra_block = ""
    if sample_extra_lines:
        sample_extra_block = "\n" + "\n".join(sample_extra_lines)

    variable_dict_block = "\n".join(variable_dict_lines)

    return f"""
fitterEngineConfig:
  likelihoodInterfaceConfig:
    jointProbabilityConfig:
      type: PoissonLLH

    dataSetList:
      - name: "TestSample"
        isEnabled: true
        model:
          tree: tree_mc
          filePathList:
            - "{root_path}"
          selectionCutFormula: "{selection_cut_formula}"
          nominalTreeWeightFormula: "{nominal_weight_formula}"
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
                  edges: [ -1, 0, 1 ]{sample_extra_block}
            dataSets: [ "TestSample" ]
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
) -> subprocess.CompletedProcess[str]:
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

    for field_name, case_map in FIELD_CASES.items():
        for case_name in case_map:
            expected_return_code = EXPECTED_RETURN_CODES[(field_name, case_name)]
            result = run_case_subprocess(field_name, case_name, script_path, script_dir, root_path)

            print(f"{field_name}/{case_name}: return code = {result.returncode}")
            if result.stdout:
                print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
            if result.stderr:
                print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")

            if result.returncode != expected_return_code:
                print(
                    f"FAIL: case {field_name}/{case_name} returned {result.returncode}, expected {expected_return_code}"
                )
                return 1

    print("SUCCESS: formula parsing matrix matches the targeted GUNDAM behavior for all tested fields.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exception:
        print(f"FAIL: {exception}")
        sys.exit(1)
