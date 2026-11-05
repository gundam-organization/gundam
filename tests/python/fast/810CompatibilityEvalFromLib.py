#!/usr/bin/env python3
"""Historical variablesTransform inputs match variableDict/evalFromLib results."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))
from EvalFromLib import (compile_shared_library, extract_child_result, print_summary_table,
                         run_case_subprocess, run_child_case, write_input_root_file,
                         write_transform_source_file)

CASE_DEFINITIONS = {"variableDict/evalFromLib": "new", "variablesTransform": "deprecated"}

def main() -> int:
    script_path = Path(__file__).resolve()
    script_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else script_path.parent

    if "--child" in sys.argv:
        syntax_mode = sys.argv[sys.argv.index("--syntax") + 1]
        root_path = Path(sys.argv[sys.argv.index("--root-file") + 1]).resolve()
        library_path = Path(sys.argv[sys.argv.index("--library-file") + 1]).resolve()
        return run_child_case(Path.cwd(), root_path, library_path, syntax_mode)

    work_dir = Path.cwd()
    root_path = work_dir / "810CompatibilityEvalFromLib.root"
    source_path = work_dir / "810CompatibilityEvalFromLib.cpp"
    library_path = work_dir / "810CompatibilityEvalFromLib.so"

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
