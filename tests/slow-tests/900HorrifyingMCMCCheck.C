#!/bin/bash
# Wrap a ROOT macro as a script.
#
#  Check the output of GUNDAM 200HorrifyingMCMC.sh against the
#  expected values.
#
root -b -n <<EOF
#include <iostream>
#include <string>
#include <memory>
#include <cmath>

#include <TFile.h>
#include <TH1F.h>
#include <TTree.h>

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
    std::shared_ptr<TFile> mcmc(new TFile("200HorrifyingMCMC.root","old"));

    TTree* tree
        = dynamic_cast<TTree*>(mcmc->Get(
                                 "FitterEngine"
                                 "/fit"
                                 "/MCMC"));
    std::vector<double> points;
    std::vector<double>* addrPoints = &points;
    tree->SetBranchAddress("Points",&addrPoints);

    TH1F normA("normA", "The first parameter",100, -1.0, 3.0);
    TH1F normB("normB", "The second parameter",100, -1.0, 3.0);
    TH1F sumAB("sumAB", "Sum of first and second parameters", 100, 1.95, 2.05);
    TH1F diffAB("diffAB", "Difference of first and second parameters",
                100, -2.5, 2.5);

    for (int i=30000; i < tree->GetEntries(); ++i) {
        tree->GetEntry(i);
        normA.Fill(points[0]);
        normB.Fill(points[1]);
        sumAB.Fill(points[0]+points[1]);
        diffAB.Fill(points[0]-points[1]);
    }

    normA.Draw();
    gPad->Print("900HorrifyingMCMCCheck.pdf(");
    normB.Draw();
    gPad->Print("900HorrifyingMCMCCheck.pdf");
    sumAB.Draw();
    gPad->Print("900HorrifyingMCMCCheck.pdf");
    diffAB.Draw();
    gPad->Print("900HorrifyingMCMCCheck.pdf)");

    // Expected accuracy.  This is based on assuming an assumed effective
    // sample size factor of 100 (i.e. it takes that many steps to become
    // uncorrelated, that's probably optimisitic).
    double entries = normA.GetEntries();
    double assumedEffectiveSampleFactor = 100;
    double relativeSigma = 1.0/std::sqrt(entries/assumedEffectiveSampleFactor);

    // The expected values are for the data generated by
    // 100NormalizationTree.C.  They need to be changed if that tree is
    // changed.
    double allowedSigma = 5.0;
    TOLERANCE("Check MCMC matches expected value for #0_Positive_C",
              normA.GetMean(),
              1.0, allowedSigma*relativeSigma);
    TOLERANCE("Check MCMC RMS matches expected value for #0_Positive_C",
              normA.GetRMS(),
              0.577, allowedSigma*relativeSigma);
    TOLERANCE("Check MCMC matches expected value for #1_Negative_C",
              normB.GetMean(),
              1.0, allowedSigma*relativeSigma);
    TOLERANCE("Check MCMC RMS matches expected value for #1_Negative_C",
              normB.GetRMS(),
              0.577, allowedSigma*relativeSigma);
    TOLERANCE("Check the sum matchs expected value",
              sumAB.GetMean(), 2.0, allowedSigma*relativeSigma);
    TOLERANCE("Check the difference matchs expected value",
              diffAB.GetMean(), 0.0, allowedSigma*relativeSigma);
    TOLERANCE("Check the difference RMS matchs expected value",
              diffAB.GetRMS(), 1.154, allowedSigma*relativeSigma);

    mcmc->Close();

    return status;
}
exit(main());
EOF
# Local Variables:
# mode:c++
# c-basic-offset:4
# End:
