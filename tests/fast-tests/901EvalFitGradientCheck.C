#!/bin/bash
# Wrap a ROOT macro as a script.
#
# Check MinimizerBase::evalFitGradient against a centered finite difference.

root -b -n <<EOF
R__LOAD_LIBRARY(libGundamFitter)

#include "FitterEngine.h"
#include "ConfigUtils.h"
#include "GundamGlobals.h"

#include "TFile.h"
#include "TDirectory.h"
#include "TSystem.h"

#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <cmath>

int status{0};
std::string args{"$*"};

#define EXPECT(msg,v1)                                      \
    do {                                                    \
        if (not (v1)) {                                     \
            std::cout << "FAIL:";                           \
            ++status;                                       \
        } else {                                            \
            std::cout << "SUCCESS:";                        \
        }                                                   \
        std::cout << " " << msg                             \
                  << " [ (" << #v1 << ") --> " << v1 << "]" \
                  << std::endl;                             \
    } while (false)

#define TOLERANCE(msg,v1,v2,tol)                            \
    do {                                                    \
        double v = std::abs(v1);                            \
        double vv = std::abs(v2);                           \
        double d = std::abs((v1)-(v2));                     \
        double r = d/std::max(0.5*(v+vv),((tol)*(tol)));    \
        if (d < (tol)) r = d;                               \
        if (r > (tol)) {                                    \
            std::cout << "FAIL:";                           \
            ++status;                                       \
        } else {                                            \
            std::cout << "SUCCESS:";                        \
        }                                                   \
        std::cout << " " << msg                             \
                  << std::setprecision(8)                   \
                  << std::scientific                        \
                  << " (" << r << "<" << (tol) << ")"       \
                  << " [" << #v1 << "=" << (v1)             \
                  << " " << #v2 << "=" << (v2)              \
                  << " " << d << "]"                        \
                  << std::endl;                             \
    } while(false);

std::vector<Parameter*> fetchFitParameters(FitterEngine& fitter) {
    std::vector<Parameter*> out;
    for( auto& parSet : fitter.getLikelihoodInterface()
                              .getModelPropagator()
                              .getParametersManager()
                              .getParameterSetsList() ){
        for( auto& par : parSet.getEffectiveParameterList() ){
            if( par.isEnabled() and not par.isPenaltyDisabled() ){
                out.emplace_back(&par);
            }
        }
    }
    return out;
}

int main() {
    std::string configFile{args + "/200NormalizationFit-config.yaml"};

    ConfigUtils::ConfigBuilder configBuilder(configFile);
    auto config = configBuilder.getConfig();
    config["fitterEngineConfig"]["minimizerConfig"]["useNormalizedFitSpace"] = false;
    config["fitterEngineConfig"]["minimizerConfig"]["writeLlhHistory"] = false;
    config["fitterEngineConfig"]["generateSamplePlots"] = false;
    config["fitterEngineConfig"]["generateOneSigmaPlots"] = false;

    ConfigReader gundamFitterConfig(config);
    gundamFitterConfig.defineFields({{"fitterEngineConfig"}});
    auto fitterEngineConfig = gundamFitterConfig.fetchValue<ConfigReader>("fitterEngineConfig");

    GundamGlobals::setNumberOfThreads(1);
    FitterEngine::setRandomSeed(10000);

    std::shared_ptr<TFile> outputFile(new TFile("901EvalFitGradientCheck.root", "recreate"));
    EXPECT("Output file must be open", outputFile and outputFile->IsOpen());
    if( not outputFile or not outputFile->IsOpen() ){ return status; }

    auto* fitterDir = outputFile->mkdir("FitterEngine");
    EXPECT("FitterEngine output directory must exist", fitterDir != nullptr);
    if( fitterDir == nullptr ){ return status; }

    FitterEngine fitter(fitterDir);
    fitter.configure(fitterEngineConfig);
    fitter.getLikelihoodInterface().setDataType(LikelihoodInterface::DataType::RealData);
    fitter.getMinimizer().setDisableCalcError(true);
    fitter.initialize();

    auto fitParameters = fetchFitParameters(fitter);
    EXPECT("Two normalization fit parameters are expected", fitParameters.size() == 2);
    if( fitParameters.size() != 2 ){ return status; }

    std::vector<double> nominalValues;
    nominalValues.reserve(fitParameters.size());
    for( auto* par : fitParameters ){
        nominalValues.emplace_back(par->getParameterValue());
    }

    auto& minimizer = fitter.getMinimizer();
    double nominalLlh = minimizer.evalFit(nominalValues.data());
    minimizer.evalFitGradient();

    EXPECT("Nominal LLH is finite", std::isfinite(nominalLlh));
    if( not std::isfinite(nominalLlh) ){ return status; }

    std::vector<double> gradientValues;
    gradientValues.reserve(fitParameters.size());
    for( auto* par : fitParameters ){
        gradientValues.emplace_back(par->getGradient());
    }

    const double step{1E-4};
    const double tolerance{5E-3};
    for( std::size_t iPar = 0 ; iPar < fitParameters.size() ; iPar++ ){
        std::vector<double> shiftedValues{nominalValues};

        shiftedValues[iPar] = nominalValues[iPar] + step;
        double up = minimizer.evalFit(shiftedValues.data());

        shiftedValues[iPar] = nominalValues[iPar] - step;
        double down = minimizer.evalFit(shiftedValues.data());

        double finiteDiff = (up - down) / (2. * step);
        TOLERANCE(
            std::string("evalFitGradient matches finite difference for ")
                + fitParameters[iPar]->getFullTitle(),
            gradientValues[iPar], finiteDiff, tolerance
        );
    }

    minimizer.evalFit(nominalValues.data());
    outputFile->Close();

    return status;
}
exit(main());
EOF
# Local Variables:
# mode:c++
# c-basic-offset:4
# End:
