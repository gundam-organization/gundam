# !/bin/bash
# Wrap a ROOT macro as a script.
root <<EOF

#include <iostream>
#include <string>
#include <memory>
#include <cmath>

#include <TFile.h>
#include <TTree.h>
#include <TRandom.h>
#include <TClonesArray.h>
#include <TGraph.h>

std::string args{"$*"};

///////////////////////////////////////////////////////////////////////
// Generate input files to be used in tests with variables, event weight
// splines, and constraint covariances that are ENTIRELY unrelated to physics.
// This produces two trees "tree_mc" and "tree_dt".  The first tree provides
// the GUNDAM "mc" inputs, and the second tree can be used as data input. The
// random seeds are closely controlled, so that the data file should be
// identically reproduced each time the script is run.
//
// tree_mc variables:
//
//    (A, B) -- These are the analogs to the detector reconstructed values.
//              Used to fill the likelihood histograms.
//    C      -- A "truth" variable.  Check that it is never precisely zero.
//    spline_C -- An event-by-event weighting spline.
//    spline_D -- An event-by-event weighting spline.
//
//    Variable "B" is generated as a normal distribution centered at 0.0.
//    Variable "A" is generated as two normal distributions.  When "C" is less
//    than or equal to zero, the distribution is centered at -1.0.  When "C"
//    is positive, the distribution is centered at 1.0.  The "spline_C"
//    weighting applies to A when "C" is greater than zero, and "spline_D"
//    applies to A when "C" is less than zero.
//
// tree_dt variables:
//
//    (A, B) -- Analogs to the detector reconstructed values.
//
//    Variable "B" is normal and centered at zero.  Variable "A" is two
//    sub-samples with the first centered at -1.0, and the second centered at
//    1.0.  The normalization for the first sub-sample is 80% of the same set
//    in tree_mc.  The normalization of the second sub-sample is 60% of the
//    same set in tree_mc.
//
// A covariance matrix is defined for the fit parameters.  The fit parameters
// are:
//
//    norm_A -- The normalization when C is greater than zero.
//    norm_B -- The normalizaiton with C is less than zero.
//    spline_C -- An event weight spline for C greater than zero.
//    spline_D -- An event weight spline for C less than zero.

///////////////////////////////////////////////////////////////////////////////
/// A class to generate a repeatable sequence of "random" Gaussian distributed
/// value
///////////////////////////////////////////////////////////////////////////////
class Gaussian {
public:
    /// Create with an explicit seed so that the sequence is repeatable.
    explicit Gaussian(int seed) {
        _generator_ = std::shared_ptr<TRandom>(new TRandom(seed));
    }

    /// Make a new value in the sequence
    double operator() () {
        _value_ = _generator_->Gaus();
        return _value_;
    }

    /// Get the last value generated in the sequence.
    double Get() {return _value_;}

private:
    /// A generator that is unique to this object.
    std::shared_ptr<TRandom> _generator_{nullptr};

    /// The last value.
    double _value_{std::nan("unset")};
};

// A class to fill a graph with an exponential function.
class ExponentialGraph {
public:
    /// Create a generator with a particular exponential scale factor.
    explicit ExponentialGraph(double s) {
        _scale_ = s;
    }

    /// Fill the graph.
    void operator()(TObject* obj,double x) {
        TGraph* graph = dynamic_cast<TGraph*>(obj);
        int n = graph->GetN();
        for (int i=0; i<n; ++i) {
            double d = i-n/2;
            double s = std::exp(x*_scale_*d);
            graph->SetPoint(i,d,s);
        }
    }

private:
    double _scale_{std::nan("unset")};
};

void writeMCTree() {
    TTree* tree = new TTree("tree_mc","GUNDAM MC Sample Tree with splines");

    Gaussian MakeA(10000);
    double varA;
    tree->Branch("A",&varA);

    Gaussian MakeB(10001);
    double varB;
    tree->Branch("B",&varB);

    Gaussian MakeC(10002);
    double varC;
    tree->Branch("C",&varC);

    ExponentialGraph fillC(0.9);
    TClonesArray spline_C("TGraph",1);
    new(spline_C[0]) TGraph(7);
    tree->Branch("spline_C",&spline_C,32000,0);

    ExponentialGraph fillD(-0.9);
    TClonesArray spline_D("TGraph",1);
    new(spline_D[0]) TGraph(7);
    tree->Branch("spline_D",&spline_D,32000,0);

    // Make events shifted by A of +1.  These events are reweighted by
    // spline_C in the fit, and spline_D is set to be flat.
    for (int i=0; i<5000; ++i) {
        do {varA = MakeA();} while (std::abs(varA) > 5.0);
        do {varB = MakeB();} while (std::abs(varB) > 5.0);
        do {varC = MakeC();} while (std::abs(varC) < 0.01);
        if (varC <= 0.0) varC = - varC;
        varA += 1.0;
        fillC(spline_C[0],varB);
        fillD(spline_D[0],0.0);
        tree->Fill();
    }

    // Make events shifted by A of -1.  These events are reweighted by
    // spline_D in the fit and spline_C is set to be flat.
    for (int i=0; i<5000; ++i) {
        do {varA = MakeA();} while (std::abs(varA) > 5.0);
        do {varB = MakeB();} while (std::abs(varB) > 5.0);
        do {varC = MakeC();} while (std::abs(varC) < 0.01);
        if (varC > 0.0) varC = - varC;
        varA -= 1.0;
        fillC(spline_C[0],0.0);
        fillD(spline_D[0],varB);
        tree->Fill();
    }

}

void writeDataTree() {
    TTree* tree = new TTree("tree_dt","GUNDAM Data Sample Tree");

    Gaussian MakeA(20000);
    double varA;
    tree->Branch("A",&varA);

    Gaussian MakeB(20001);
    double varB;
    tree->Branch("B",&varB);

    // Write a data sample for "C > 0" that is shifted by A of +1.
    for (int i=0; i<5000; ++i) {
        do {varA = MakeA();} while (std::abs(varA) > 5.0);
        do {varB = MakeB();} while (std::abs(varB) > 5.0);
        varA += 1.0;
        tree->Fill();
    }

    // Write a data sample for "C < 0" that is shifted by A of -1.
    for (int i=0; i<5000; ++i) {
        do {varA = MakeA();} while (std::abs(varA) > 5.0);
        do {varB = MakeB();} while (std::abs(varB) > 5.0);
        varA -= 1.0;
        tree->Fill();
    }
}

void writeCovariance() {
    TMatrixDSym cov(4);
    double sigma = 0.3;
    double correlation = 0.5;
    for (int i=0; i<cov.GetNrows(); ++i) {
        for (int j=0; j<cov.GetNrows(); ++j) {
            if (i == j) {
                cov(i,j) = sigma*sigma;
                continue;
            }
            double covar = correlation*sigma*sigma;
            double d = std::pow(-1.0,i+j);
            if (d>0.0) cov(i,j) = covar;
            else cov(i,j) = -covar;
        }
    }
    cov.Write("CovarianceInputCovariance");

    TVectorD priors(4);
    priors[0] = 1.0;
    priors[1] = 1.0;
    priors[2] = 0.0;
    priors[3] = 0.0;
    priors.Write("CovarianceInputPriors");

    TObjArray names;
    names.SetOwner(true);
    names.AddLast(new TObjString("norm_A"));
    names.AddLast(new TObjString("norm_B"));
    names.AddLast(new TObjString("spline_C"));
    names.AddLast(new TObjString("spline_D"));
    names.Write("CovarianceInputNames",1);

}

int main() {
    std::shared_ptr<TFile> file(new TFile("100CovarianceTree.root","new"));
    if (!file || !file->IsOpen()) {
        std::cout << "FAIL: TFile not opened" << std::endl;
        return 1;
    }

    writeMCTree();
    writeDataTree();
    writeCovariance();

    file->Write();
    file->Close();

    return 0;
}
exit(main());
EOF
# Local Variables:
# mode:c++
# c-basic-offset:4
# End:
