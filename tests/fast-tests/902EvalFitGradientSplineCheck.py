#!/usr/bin/env python3

import math
import os
from pathlib import Path
import sys

import GUNDAM


STATUS = 0


def expect(message, value):
    global STATUS
    if value:
        print(f"SUCCESS: {message} [ {value} ]")
    else:
        print(f"FAIL: {message} [ {value} ]")
        STATUS += 1


def tolerance(message, value, reference, tolerance_value):
    global STATUS
    delta = abs(value - reference)
    scale = max(0.5 * (abs(value) + abs(reference)), tolerance_value * tolerance_value)
    residual = delta / scale
    if delta < tolerance_value:
        residual = delta

    if residual > tolerance_value:
        print(
            "FAIL: "
            f"{message} ({residual:.8e}<{tolerance_value:.8e}) "
            f"[gradient={value:.8e} finiteDiff={reference:.8e} delta={delta:.8e}]"
        )
        STATUS += 1
    else:
        print(
            "SUCCESS: "
            f"{message} ({residual:.8e}<{tolerance_value:.8e}) "
            f"[gradient={value:.8e} finiteDiff={reference:.8e} delta={delta:.8e}]"
        )


def make_fitter(script_dir):
    config_builder = GUNDAM.ConfigUtils.ConfigBuilder(
        f"{script_dir}/200CovarianceFit-config.yaml"
    )
    override_path = Path("902EvalFitGradientSplineCheck-override.yaml")
    override_path.write_text(
        "\n".join(
            [
                "fitterEngineConfig:",
                "  generateSamplePlots: false",
                "  generateOneSigmaPlots: false",
                "  minimizerConfig:",
                "    useNormalizedFitSpace: false",
                "    writeLlhHistory: false",
                "",
            ]
        )
    )
    config_builder.override(str(override_path))

    config_reader = GUNDAM.ConfigUtils.ConfigReader(config_builder.getConfig())
    config_reader.defineField(
        GUNDAM.ConfigUtils.ConfigReader.FieldDefinition("fitterEngineConfig")
    )

    app = GUNDAM.GundamApp("902EvalFitGradientSplineCheck")
    app.openOutputFile("902EvalFitGradientSplineCheck.root")

    fitter = GUNDAM.FitterEngine()
    fitter.setSaveDir(app, "FitterEngine")
    fitter.configure(config_reader.fetchValueConfigReader("fitterEngineConfig"))
    fitter.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.RealData)
    fitter.getMinimizer().setDisableCalcError(True)
    fitter.initialize()

    return app, fitter


def is_spline_parameter(parameter):
    return parameter.getName() in ["spline_C", "spline_D"]


def main():
    script_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.environ["CONFIG_DIR"] = script_dir
    os.environ["DATA_DIR"] = os.getcwd()

    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)
    GUNDAM.FitterEngine.setRandomSeed(10000)

    app, fitter = make_fitter(script_dir)

    minimizer = fitter.getMinimizer()
    fit_parameters = minimizer.getMinimizerFitParameterList()

    expect("Four covariance fit parameters are expected", len(fit_parameters) == 4)
    if len(fit_parameters) != 4:
        return STATUS

    spline_parameter_indices = []
    spline_parameters = []
    for parameter_index, parameter in enumerate(fit_parameters):
        if not is_spline_parameter(parameter):
            continue
        spline_parameter_indices.append(parameter_index)
        spline_parameters.append(parameter)

    expect("Two spline fit parameters are expected", len(spline_parameters) == 2)
    if len(spline_parameters) != 2:
        return STATUS

    nominal_values = [parameter.getParameterValue() for parameter in fit_parameters]
    nominal_llh = minimizer.evalFit(nominal_values)
    minimizer.evalFitGradient(spline_parameters)

    expect("Nominal LLH is finite", math.isfinite(nominal_llh))
    if not math.isfinite(nominal_llh):
        return STATUS

    gradient_values = [parameter.getGradient() for parameter in spline_parameters]

    step = 1.0e-4
    tolerance_value = 1.0e-2
    for spline_index, parameter in enumerate(spline_parameters):
        parameter_index = spline_parameter_indices[spline_index]

        shifted_values = list(nominal_values)
        shifted_values[parameter_index] = nominal_values[parameter_index] + step
        up = minimizer.evalFit(shifted_values)

        shifted_values[parameter_index] = nominal_values[parameter_index] - step
        down = minimizer.evalFit(shifted_values)

        finite_diff = (up - down) / (2.0 * step)
        tolerance(
            "evalFitGradient matches finite difference for spline parameter "
            + parameter.getFullTitle(),
            gradient_values[spline_index],
            finite_diff,
            tolerance_value,
        )

    minimizer.evalFit(nominal_values)
    return STATUS


if __name__ == "__main__":
    sys.exit(main())
