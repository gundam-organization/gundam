#!/bin/bash
# Wrap a ROOT macro as a script.
#
#  Check the output of GUNDAM 200DecompositionFit.sh against the
#  expected values.
#
root -b -n <<EOF
#include <iostream>
#include <string>
#include <memory>
#include <cmath>

#include <TFile.h>
#include <TH1.h>

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
#define ABSOLUTE_TOLERANCE(msg,v1,v2,tol)                   \
    do {                                                    \
        double v = (v1)>0 ? (v1): -(v1);                    \
        double vv = (v2)>0 ? (v2): -(v2);                   \
        double d = std::abs((v1)-(v2));                     \
        if (d > (tol)) {                                    \
            std::cout << "FAIL:";                           \
            ++ status;                                      \
        } else {                                            \
            std::cout << "SUCCESS:";                        \
        }                                                   \
        std::cout << " " << msg                             \
                  << std::setprecision(8)                   \
                  << std::scientific                        \
                  << " (" << d << "<" << (tol) << ")"       \
                  << " [" << #v1 << "=" << (v1)             \
                  << " " << #v2 << "=" << (v2) << "]"       \
                  << std::endl;                             \
    } while(false);

int main() {
    std::shared_ptr<TFile> file(new TFile("200DecompositionFit.root","old"));
    std::shared_ptr<TFile> target(new TFile("200CovarianceFit.root","old"));

    EXPECT("File pointer is not null",file);
    if (!file) return status;

    EXPECT("File must be open", file->IsOpen());
    if (not file->IsOpen()) return status;

    TH1* postFitErrorsMigrad
        = dynamic_cast<TH1*>(file->Get(
                                 "FitterEngine"
                                 "/postFit"
                                 "/Migrad"
                                 "/errors"
                                 "/CovarianceConstraints"
                                 "/values"
                                 "/postFitErrors_TH1D"));
    EXPECT("postFitErrors must exist",  postFitErrorsMigrad);

    TH1* postFitErrorsHesse
        = dynamic_cast<TH1*>(file->Get(
                                 "FitterEngine"
                                 "/postFit"
                                 "/Hesse"
                                 "/errors"
                                 "/CovarianceConstraints"
                                 "/values"
                                 "/postFitErrors_TH1D"));
    EXPECT("postFitErrors must exist",  postFitErrorsHesse);

    TMatrixD* covariance
        = dynamic_cast<TMatrixD*>(file->Get(
                                      "FitterEngine"
                                      "/postFit"
                                      "/Hesse"
                                      "/errors"
                                      "/CovarianceConstraints"
                                      "/matrices"
                                      "/Covariance_TMatrixD"));
    EXPECT("covariance must exist",  covariance);

    TH1* targetHesse
        = dynamic_cast<TH1*>(target->Get(
                                 "FitterEngine"
                                 "/postFit"
                                 "/Hesse"
                                 "/errors"
                                 "/CovarianceConstraints"
                                 "/values"
                                 "/postFitErrors_TH1D"));
    EXPECT("targetFitErrors must exist",  targetHesse);

    TMatrixD* targetCov
        = dynamic_cast<TMatrixD*>(target->Get(
                                      "FitterEngine"
                                      "/postFit"
                                      "/Hesse"
                                      "/errors"
                                      "/CovarianceConstraints"
                                      "/matrices"
                                      "/Covariance_TMatrixD"));
    EXPECT("target covariance must exist",  targetCov);

    // Don't try to continue if the data is missing from the file.
    if (not postFitErrorsMigrad) return status;
    if (not postFitErrorsHesse) return status;
    if (not covariance) return status;

    postFitErrorsHesse->Draw("E");
    gPad->Print("900DecompositionFitCheck.pdf");

    covariance->Print();

    // Change this to set the expected relative tolerance.
    double tolerance = 5E-6;

    // The expected values are for the data generated by
    // 100CovarianceTree.C.  They need to be changed if that tree is
    // changed.
    ABSOLUTE_TOLERANCE("Check HESSE value for #0_norm_A",
                       postFitErrorsHesse->GetBinContent(1),
                       targetHesse->GetBinContent(1), tolerance);
    ABSOLUTE_TOLERANCE("Check error for #0_norm_A",
                       postFitErrorsHesse->GetBinError(1),
                       targetHesse->GetBinError(1), tolerance);

    ABSOLUTE_TOLERANCE("Check HESSE value for #1_norm_B",
                       postFitErrorsHesse->GetBinContent(2),
                       targetHesse->GetBinContent(2), tolerance);
    ABSOLUTE_TOLERANCE("Check error for #1_norm_B",
                       postFitErrorsHesse->GetBinError(2),
                       targetHesse->GetBinError(2), tolerance);

    ABSOLUTE_TOLERANCE("Check HESSE value for #2_spline_C",
                       postFitErrorsHesse->GetBinContent(3),
                       targetHesse->GetBinContent(3),tolerance);
    ABSOLUTE_TOLERANCE("Check error for #2_spline_C",
                       postFitErrorsHesse->GetBinError(3),
                       targetHesse->GetBinError(3), tolerance);

    ABSOLUTE_TOLERANCE("Check HESSE value for #3_spline_D",
                       postFitErrorsHesse->GetBinContent(4),
                       targetHesse->GetBinContent(4), tolerance);
    ABSOLUTE_TOLERANCE("Check error for #3_spline_D",
                       postFitErrorsHesse->GetBinError(4),
                       targetHesse->GetBinError(4), tolerance);

    file->Close();

    return status;
}
exit(main());
EOF
# Local Variables:
# mode:c++
# c-basic-offset:4
# End:
