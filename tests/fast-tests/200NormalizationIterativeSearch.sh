#!/bin/bash

# Set the base name for this test (should match the script name)
BASE=200NormalizationIterativeSearch

# Get the directory containing the script from the command line
# parameters (avoids bash trickery).  Use the current directory as the
# default.
DIR=.
if [ ${#1} -gt 0 ]; then
    DIR=${1}
fi

# Make sure that gundam has been setup.
if ! which gundamFitter; then
    echo FAIL: Executable not found for gundamFitter
    exit 1
fi

# Set the expected locations for the config and output files.
export CONFIG_DIR=${DIR}
export DATA_DIR=${PWD}

CONFIG_FILE=${CONFIG_DIR}/${BASE}-config.yaml
OUTPUT_FILE=${DATA_DIR}/${BASE}.root

echo ${OUTPUT_FILE}
echo ${CONFIG_FILE}

if ! gundamFitter --debug --cpu -t 1 -s 10000 -c ${CONFIG_FILE} -o ${OUTPUT_FILE}; then
    echo FAIL: gundamFitter did not finish
    exit 1
fi

# Check the search output. Positive_C is profiled over 0.4, 0.5, 0.6, 0.7, 0.8
# and the data was generated with Positive_C = 0.6 and Negative_C = 0.8.
root -b -q -l <<ROOTEOF
{
    TFile file("${OUTPUT_FILE}");
    TTree* pointList = (TTree*) file.Get("FitterEngine/postFit/pointList");
    TTree* problemDefinition = (TTree*) file.Get("FitterEngine/postFit/problemDefinition");
    if (not pointList or not problemDefinition) {
        std::cout << "FAIL: pointList or problemDefinition is missing" << std::endl;
        return 1;
    }
    if (pointList->GetEntries() != 5) {
        std::cout << "FAIL: expected 5 points, got " << pointList->GetEntries() << std::endl;
        return 1;
    }

    int point; double llh; bool converged; double profiledValues[1]; double parValues[1];
    pointList->SetBranchAddress("Point", &point);
    pointList->SetBranchAddress("LLH", &llh);
    pointList->SetBranchAddress("Converged", &converged);
    pointList->SetBranchAddress("ProfiledValues", profiledValues);
    pointList->SetBranchAddress("PostFitParameterValues", parValues);

    double bestLlh = 1E300; double bestPositive = -1; double bestNegative = -1;
    for (int i = 0; i < pointList->GetEntries(); ++i) {
        pointList->GetEntry(i);
        std::cout << "point " << point << ": Positive_C = " << profiledValues[0]
                  << " Negative_C = " << parValues[0] << " llh = " << llh
                  << (converged ? "" : " NOT CONVERGED") << std::endl;
        if (not converged) { std::cout << "FAIL: point " << point << " did not converge" << std::endl; return 1; }
        if (llh < bestLlh) { bestLlh = llh; bestPositive = profiledValues[0]; bestNegative = parValues[0]; }
    }

    int nPoints, nProfiledParameters, nParameters; double bestLlhInFile;
    problemDefinition->SetBranchAddress("nPoints", &nPoints);
    problemDefinition->SetBranchAddress("nProfiledParameters", &nProfiledParameters);
    problemDefinition->SetBranchAddress("nParameters", &nParameters);
    problemDefinition->SetBranchAddress("BestLLHInFile", &bestLlhInFile);
    problemDefinition->GetEntry(0);
    if (nPoints != 5 or nProfiledParameters != 1 or nParameters != 1) {
        std::cout << "FAIL: problemDefinition says " << nPoints << " points, " << nProfiledParameters
                  << " profiled parameters, " << nParameters << " parameters" << std::endl;
        return 1;
    }
    if (bestLlhInFile != bestLlh) {
        std::cout << "FAIL: BestLLHInFile " << bestLlhInFile << " differs from the pointList minimum " << bestLlh << std::endl;
        return 1;
    }

    if (std::abs(bestPositive - 0.6) > 1E-9) {
        std::cout << "FAIL: best point has Positive_C = " << bestPositive << ", expected 0.6" << std::endl;
        return 1;
    }
    if (std::abs(bestNegative - 0.8) > 0.05) {
        std::cout << "FAIL: Negative_C at the best point is " << bestNegative << ", expected 0.8" << std::endl;
        return 1;
    }
    std::cout << "SUCCESS: iterative search found Positive_C = " << bestPositive << ", Negative_C = " << bestNegative << std::endl;
    return 0;
}
ROOTEOF
if [ $? -ne 0 ]; then
    echo FAIL: output check failed
    exit 1
fi

# End of the script
