#!/bin/bash
# Wrap a ROOT macro as a script.
#
#  Check the output of GUNDAM 200NormalizationAsimov.sh against the
#  expected values.
#
root -b -n <<EOF
#include <iostream>
#include <string>
#include <memory>
#include <cmath>

#include <TFile.h>
#include <TH1.h>
#include <TKey.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

std::string args{"$*"};
int status{0};

/// Fail with message if "v1" evaluates to false.  THIS IS COPIED
/// HERE TO AVOID DEPENDENCIES
#define EXPECT(msg,v1)                                      \
    do {                                                    \
        if (not (v1)) {                                     \
            std::cout << "FAIL:";                           \
            ++ status;                                      \
        } else {                                            \
            std::cout << "SUCCESS:";                        \
        }                                                   \
        std::cout << " " << msg                             \
                  << " [ (" << #v1 << ") --> " << v1 << "]" \
                  << std::endl;                             \
    } while (false)

/// Fail if fractional difference between "v1" and "v2" is larger than "tol"
/// THIS IS COPIED HERE TO AVOID DEPENDENCIES
#define TOLERANCE(msg,v1,v2,tol)                            \
    do {                                                    \
        double v = (v1)>0 ? (v1): -(v1);                    \
        double vv = (v2)>0 ? (v2): -(v2);                   \
        double d = std::abs((v1)-(v2));                     \
        double r = d/std::max(0.5*(v+vv),(tol));            \
        if (r > (tol)) {                                    \
            std::cout << "FAIL:";                           \
            ++ status;                                      \
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

int main() {
    std::shared_ptr<TFile> file(new TFile("200NormalizationAsimov.root","old"));

    EXPECT("File pointer is not null",file);
    if (!file) return status;

    EXPECT("File must be open", file->IsOpen());
    if (not file->IsOpen()) return status;

    auto* runtimeDir = file->GetDirectory("gundam/runtime");
    EXPECT("Runtime directory exists", runtimeDir);
    if (runtimeDir == nullptr) return status;
    int loggerKeys = 0;
    for (auto* key : *runtimeDir->GetListOfKeys()) {
        if (std::string(key->GetName()) == "logger") ++loggerKeys;
    }
    EXPECT("Logger tree has one key", (loggerKeys == 1));

    TTreeReader logReader("gundam/runtime/logger", file.get());
    TTreeReaderValue<std::string> line(logReader, "text");
    bool createdOutput = false;
    bool startedFit = false;
    bool closedOutput = false;
    while (logReader.Next()) {
        createdOutput |= line->find("Creating output file:") != std::string::npos;
        startedFit |= line->find("Running the fitter...") != std::string::npos;
        closedOutput |= line->find("Closing output file") != std::string::npos;
    }
    EXPECT("Logger output includes the output file creation", createdOutput);
    EXPECT("Logger output includes the start of the fit", startedFit);
    EXPECT("Logger output includes the end of the run", closedOutput);

    TH1* postFitErrorsMigrad
        = dynamic_cast<TH1*>(file->Get(
                                 "FitterEngine"
                                 "/postFit"
                                 "/Migrad"
                                 "/errors"
                                 "/Normalizations"
                                 "/values"
                                 "/postFitErrors_TH1D"));
    EXPECT("postFitErrors must exist",  postFitErrorsMigrad);

    TH1* postFitErrorsHesse
        = dynamic_cast<TH1*>(file->Get(
                                 "FitterEngine"
                                 "/postFit"
                                 "/Hesse"
                                 "/errors"
                                 "/Normalizations"
                                 "/values"
                                 "/postFitErrors_TH1D"));
    EXPECT("postFitErrors must exist",  postFitErrorsHesse);

    TMatrixD* covariance
        = dynamic_cast<TMatrixD*>(file->Get(
                                      "FitterEngine"
                                      "/postFit"
                                      "/Hesse"
                                      "/errors"
                                      "/Normalizations"
                                      "/matrices"
                                      "/Covariance_TMatrixD"));
    EXPECT("covariance must exist",  covariance);

    // Don't try to continue if the data is missing from the file.
    if (not postFitErrorsMigrad) return status;
    if (not postFitErrorsHesse) return status;
    if (not covariance) return status;

    postFitErrorsMigrad->Draw();
    gPad->Print("900NormalizationAsimovCheck.pdf[");
    postFitErrorsHesse->Draw();
    gPad->Print("900NormalizationAsimovCheck.pdf");
    gPad->Print("900NormalizationAsimovCheck.pdf]");

    covariance->Print();

    // Change this to set the expected absolute tolerance.
    double tolerance = 1E-6;

    // The expected values are for the data generated by
    // 100NormalizationTree.C.  They need to be changed if that tree is
    // changed.
    TOLERANCE("Check MIGRAD value for #0_Positive_C",
              postFitErrorsMigrad->GetBinContent(1), 1.0, tolerance);
    TOLERANCE("Check HESSE value for #0_Positive_C",
              postFitErrorsHesse->GetBinContent(1), 1.0, tolerance);
    TOLERANCE("Check variance for #0_Positive_C",
              (*covariance)(0,0), 1.455e-04, 0.001);

    TOLERANCE("Check MIGRAD value for #1_Negative_C",
              postFitErrorsMigrad->GetBinContent(2), 1.0, tolerance);
    TOLERANCE("Check HESSE value for #1_Negative_C",
              postFitErrorsHesse->GetBinContent(2), 1.0, tolerance);
    TOLERANCE("Check variance for #1_Negative_C",
              (*covariance)(1,1), 1.455e-04, 0.001);

    TOLERANCE("Check covariance",
              (*covariance)(0,1), -4.548e-05, 0.001);

    file->Close();

    return status;
}
exit(main());
EOF
# Local Variables:
# mode:c++
# c-basic-offset:4
# End:
