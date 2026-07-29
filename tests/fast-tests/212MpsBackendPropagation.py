#!/usr/bin/env python3

import importlib.util
import math
import sys
from pathlib import Path


def load_cpu_backend_test_module():
    test_path = Path(__file__).resolve().with_name("211CpuBackendPropagation.py")
    spec = importlib.util.spec_from_file_location("cpu_backend_test", test_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_mps_config_text(cpu_test_module, root_path: Path, enable_backend: bool) -> str:
    config_text = cpu_test_module.build_config_text(root_path, enable_backend)
    if enable_backend:
        config_text = config_text.replace("type: CPU", "type: MPS")
        config_text = config_text.replace(
            "outputRequests: [EventWeights, Histograms]",
            "outputRequests: [EventWeights, Histograms, Likelihood]",
        )
    return config_text


def assert_mps_close_list(label: str, left: list[float], right: list[float]) -> None:
    if len(left) != len(right):
        raise RuntimeError(f"{label}: length mismatch {len(left)} != {len(right)}")

    for i, (left_value, right_value) in enumerate(zip(left, right)):
        if not math.isclose(left_value, right_value, rel_tol=1e-5, abs_tol=1e-5):
            raise RuntimeError(f"{label}: bin {i} mismatch {left_value} != {right_value}")


def main() -> int:
    cpu_test = load_cpu_backend_test_module()

    work_dir = Path.cwd()
    root_path = work_dir / "212MpsBackendPropagation.root"

    cpu_test.write_input_root_file(root_path)

    prior_standard_llh, prior_standard_sums, prior_standard_errors = cpu_test.evaluate_config(
        build_mps_config_text(cpu_test, root_path, False),
        work_dir,
    )
    prior_backend_llh, prior_backend_sums, prior_backend_errors = cpu_test.evaluate_config(
        build_mps_config_text(cpu_test, root_path, True),
        work_dir,
    )

    systematic_point = (4.0, 1.0)
    standard_llh, standard_sums, standard_errors = cpu_test.evaluate_config(
        build_mps_config_text(cpu_test, root_path, False),
        work_dir,
        systematic_point,
    )
    backend_llh, backend_sums, backend_errors = cpu_test.evaluate_config(
        build_mps_config_text(cpu_test, root_path, True),
        work_dir,
        systematic_point,
    )

    expected_prior_sums = [7.0, 8.0]
    expected_shifted_sums = [9.0, 6.0]
    cpu_test.assert_close_list("standard prior sums", prior_standard_sums, expected_prior_sums)
    assert_mps_close_list("MPS prior sums", prior_backend_sums, prior_standard_sums)
    assert_mps_close_list("MPS prior errors", prior_backend_errors, prior_standard_errors)

    cpu_test.assert_close_list("standard shifted sums", standard_sums, expected_shifted_sums)
    assert_mps_close_list("MPS shifted sums", backend_sums, standard_sums)
    assert_mps_close_list("MPS shifted errors", backend_errors, standard_errors)

    if not math.isclose(prior_backend_llh, prior_standard_llh, rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError(f"Prior LLH mismatch {prior_backend_llh} != {prior_standard_llh}")

    if not math.isclose(backend_llh, standard_llh, rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError(f"Shifted LLH mismatch {backend_llh} != {standard_llh}")

    if math.isclose(standard_llh, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Shifted LLH is unexpectedly zero.")

    print("Prior standard bin sums:", prior_standard_sums)
    print("Prior MPS bin sums:", prior_backend_sums)
    print("Shifted standard bin sums:", standard_sums)
    print("Shifted MPS bin sums:", backend_sums)
    print("Prior standard LLH:", prior_standard_llh)
    print("Prior MPS LLH:", prior_backend_llh)
    print("Shifted standard LLH:", standard_llh)
    print("Shifted MPS LLH:", backend_llh)
    print("SUCCESS: MPS backend propagation matches the standard propagation path at prior and shifted parameters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
