#!/usr/bin/env python3

import json
import shutil
import subprocess
import sys
from array import array
from pathlib import Path
from typing import Any, Dict, List


CASE_DEFINITIONS = {
    "variableDict/evalFromLib": "new",
    "variablesTransform": "deprecated",
}


def flush_python_outputs() -> None:
    sys.stdout.flush()
    sys.stderr.flush()


def format_return_code(return_code: int) -> str:
    if return_code < 0:
        return "signal {0}".format(-return_code)
    return str(return_code)


def print_summary_table(title: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    case_width = max(len("Case"), max(len(row["case"]) for row in rows))
    status_width = len("Status")
    rc_width = max(len("Return"), max(len(format_return_code(row["return_code"])) for row in rows))

    print()
    print(title)
    print(
        "{0:<{case_width}}  {1:<{status_width}}  {2:<{rc_width}}".format(
            "Case",
            "Status",
            "Return",
            case_width=case_width,
            status_width=status_width,
            rc_width=rc_width,
        )
    )
    print(
        "{0}  {1}  {2}".format(
            "-" * case_width,
            "-" * status_width,
            "-" * rc_width,
        )
    )
    for row in rows:
        print(
            "{0:<{case_width}}  {1:<{status_width}}  {2:<{rc_width}}".format(
                row["case"],
                row["status"],
                format_return_code(row["return_code"]),
                case_width=case_width,
                status_width=status_width,
                rc_width=rc_width,
            )
        )


def write_input_root_file(root_path: Path) -> None:
    import uproot

    with uproot.recreate(root_path) as root_file:
        tree = root_file.mktree("tree_mc", {"X": "float64"})
        tree.extend({"X": array("d", [0.5] * 10)})


def write_transform_source_file(source_path: Path) -> None:
    source_path.write_text(
        "\n".join(
            [
                'extern "C" double evalVariable(double* inputList){',
                "  return inputList[0] + 1.0;",
                "}",
                "",
            ]
        ),
        encoding="ascii",
    )


def compile_shared_library(source_path: Path, library_path: Path) -> None:
    compiler_path = shutil.which("g++")
    if compiler_path is None:
        raise RuntimeError("Could not find g++ in PATH.")

    compile_command = [compiler_path, "-O2", "-fPIC", str(source_path)]
    if sys.platform == "darwin":
        compile_command.extend(["-dynamiclib", "-o", str(library_path)])
    else:
        compile_command.extend(["-shared", "-o", str(library_path)])

    result = subprocess.run(
        compile_command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not compile test shared library.\nstdout:\n{0}\nstderr:\n{1}".format(
                result.stdout,
                result.stderr,
            )
        )


def build_config_text(root_path: Path, library_path: Path, syntax_mode: str) -> str:
    if syntax_mode == "new":
        transform_block = """
          variableDict:
            - name: "Y"
              evalFromLib:
                title: "YPlusOne"
                libraryFile: "{library_path}"
                inputList:
                  - "X"
""".format(
            library_path=library_path
        )
    elif syntax_mode == "deprecated":
        transform_block = """
          variablesTransform:
            - name: "YPlusOne"
              outputVariableName: "Y"
              libraryFile: "{library_path}"
              inputList:
                - "X"
""".format(
            library_path=library_path
        )
    else:
        raise RuntimeError("Unknown syntax mode: {0}".format(syntax_mode))

    return """
fitterEngineConfig:
  likelihoodInterfaceConfig:
    jointProbabilityConfig:
      type: PoissonLLH
      ignoreBinsWithZeroPredictionAtPrior: true

    dataSetList:
      - name: "TestSample"
        isEnabled: true
        model:
          allowMultipleSamplesPerEntry: true
          tree: tree_mc
          filePathList:
            - "{root_path}"
{transform_block}

    propagatorConfig:
      sampleSetConfig:
        sampleList:
          - name: Y
            isEnabled: true
            binning:
              binningDefinition:
                - name: "Y"
                  edges: [0, 1, 2]
            dataSets: ["TestSample"]
          - name: X
            isEnabled: true
            binning:
              binningDefinition:
                - name: "X"
                  edges: [0, 1, 2]
            dataSets: ["TestSample"]
""".format(
        root_path=root_path,
        transform_block=transform_block.rstrip(),
    )


def extract_sample_bin_results(sample: Any, sample_label: str) -> List[Dict[str, Any]]:
    bin_content_list = sample.getHistogram().getBinContentList()
    bin_context_list = sample.getHistogram().getBinContextList()

    if len(bin_content_list) != 2:
        raise RuntimeError(
            "Expected 2 bins for sample {0}, got {1}".format(sample_label, len(bin_content_list))
        )
    if len(bin_context_list) != 2:
        raise RuntimeError(
            "Expected 2 bin contexts for sample {0}, got {1}".format(sample_label, len(bin_context_list))
        )

    bin_results = []
    print("Bin contents ({0}/{1}):".format(sample_label, sample.getName()))
    for bin_context, bin_content in zip(bin_context_list, bin_content_list):
        bin_summary = bin_context.bin.getSummary(False)
        sum_weights = float(bin_content.sumWeights)
        print("  {0} -> sumWeights={1}".format(bin_summary, sum_weights))
        bin_results.append(
            {
                "summary": bin_summary,
                "sumWeights": sum_weights,
            }
        )

    return bin_results


def evaluate_config(config_text: str, work_dir: Path, case_label: str) -> Dict[str, Any]:
    import GUNDAM

    flush_python_outputs()

    GUNDAM.setRuntimeWorkingDirectory(str(work_dir))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)

    config_builder = GUNDAM.ConfigUtils.ConfigBuilder()
    config_builder.setConfigFromYamlString(config_text)
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

    sample_list = likelihood_interface.getModelPropagator().getSampleSet().getSampleList()
    if len(sample_list) != 2:
        raise RuntimeError("Expected 2 samples, got {0}".format(len(sample_list)))

    y_sample = sample_list[0]
    x_sample = sample_list[1]
    if y_sample.getName() != "Y":
        raise RuntimeError("Expected first sample to be Y, got {0}".format(y_sample.getName()))
    if x_sample.getName() != "X":
        raise RuntimeError("Expected second sample to be X, got {0}".format(x_sample.getName()))

    y_bin_results = extract_sample_bin_results(y_sample, case_label)
    x_bin_results = extract_sample_bin_results(x_sample, case_label)

    if y_bin_results[0]["sumWeights"] != 0.0:
        raise RuntimeError(
            "Expected Y bin [0, 1] to be empty, got {0}".format(y_bin_results[0]["sumWeights"])
        )
    if y_bin_results[1]["sumWeights"] != 10.0:
        raise RuntimeError(
            "Expected Y bin [1, 2] to contain all events, got {0}".format(y_bin_results[1]["sumWeights"])
        )
    if x_bin_results[0]["sumWeights"] != 10.0:
        raise RuntimeError(
            "Expected X bin [0, 1] to contain all events, got {0}".format(x_bin_results[0]["sumWeights"])
        )
    if x_bin_results[1]["sumWeights"] != 0.0:
        raise RuntimeError(
            "Expected X bin [1, 2] to be empty, got {0}".format(x_bin_results[1]["sumWeights"])
        )

    return {
        "case": case_label,
        "samples": {
            "Y": y_bin_results,
            "X": x_bin_results,
        },
        "likelihood": float(likelihood_interface.getLastLikelihood()),
    }


def run_case_subprocess(
    script_path: Path,
    script_dir: Path,
    root_path: Path,
    source_path: Path,
    library_path: Path,
    syntax_mode: str,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(script_dir),
            "--child",
            "--syntax",
            syntax_mode,
            "--root-file",
            str(root_path),
            "--source-file",
            str(source_path),
            "--library-file",
            str(library_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def extract_child_result(stdout_text: str) -> Dict[str, Any]:
    for line in stdout_text.splitlines():
        if line.startswith("RESULT_JSON="):
            return json.loads(line[len("RESULT_JSON="):])
    raise RuntimeError("Could not find RESULT_JSON in child stdout.")


def run_child_case(
    work_dir: Path,
    root_path: Path,
    library_path: Path,
    syntax_mode: str,
) -> int:
    config_text = build_config_text(root_path, library_path, syntax_mode)
    result_payload = evaluate_config(config_text, work_dir, syntax_mode)
    print("RESULT_JSON={0}".format(json.dumps(result_payload, sort_keys=True)))
    return 0


def main() -> int:
    script_path = Path(__file__).resolve()
    script_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else script_path.parent

    if "--child" in sys.argv:
        syntax_mode = sys.argv[sys.argv.index("--syntax") + 1]
        root_path = Path(sys.argv[sys.argv.index("--root-file") + 1]).resolve()
        library_path = Path(sys.argv[sys.argv.index("--library-file") + 1]).resolve()
        return run_child_case(Path.cwd(), root_path, library_path, syntax_mode)

    work_dir = Path.cwd()
    root_path = work_dir / "220EvalFromLibCompatibility.root"
    source_path = work_dir / "220EvalFromLibCompatibility.cpp"
    library_path = work_dir / "220EvalFromLibCompatibility.so"

    write_input_root_file(root_path)
    write_transform_source_file(source_path)
    compile_shared_library(source_path, library_path)

    summary_rows = []
    case_results = {}
    has_failure = False

    for case_name, syntax_mode in CASE_DEFINITIONS.items():
        result = run_case_subprocess(
            script_path,
            script_dir,
            root_path,
            source_path,
            library_path,
            syntax_mode,
        )

        print("{0}: return code = {1}".format(case_name, result.returncode))
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")

        status = "PASS" if result.returncode == 0 else "FAIL"
        if status == "FAIL":
            has_failure = True
        else:
            case_results[case_name] = extract_child_result(result.stdout)

        summary_rows.append(
            {
                "case": case_name,
                "status": status,
                "return_code": result.returncode,
            }
        )

    print_summary_table("Case summary", summary_rows)

    if has_failure:
        print("FAIL: one or more evalFromLib compatibility runs failed.")
        return 1

    new_result = case_results["variableDict/evalFromLib"]
    deprecated_result = case_results["variablesTransform"]

    if new_result["samples"] != deprecated_result["samples"]:
        print("FAIL: bin contents differ between variableDict/evalFromLib and variablesTransform.")
        print("  variableDict/evalFromLib: {0}".format(new_result["samples"]))
        print("  variablesTransform: {0}".format(deprecated_result["samples"]))
        return 1

    print("SUCCESS: variableDict/evalFromLib and variablesTransform produce identical bin contents.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exception:
        print("FAIL: {0}".format(exception))
        sys.exit(1)
