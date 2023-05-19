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

std::string args{"$*"};

///////////////////////////////////////////////////////////////////////
// Generate input files to be used in tests with variables that are ENTIRELY
// unrelated to physics.  This produces two trees "tree_mc" and "tree_dt".
// The first tree provides the GUNDAM "mc" inputs, and the second tree can be
// used as data input. The random seeds are closely controlled, so that the
// data file should be identically reproduced each time the script is run.
//
// tree_mc variables:
//
//    (A, B) -- These are the analogs to the detector reconstructed values.
//              Used to fill the likelihood histograms.
//    C      -- A "truth" variable.
//
//    Variable "B" is generated as a normal distribution centered at 0.0.
//    Variable "A" is generated as two normal distributions.  When "C" is less
//    than or equal to zero, the distribution is centered at -1.0.  When "C"
//    is positive, the distribution is centered at 1.0.
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

///////////////////////////////////////////////////////////////////////////////
/// A class to generate a repeatable sequence of "random" Gaussian distributed
/// value
///////////////////////////////////////////////////////////////////////////////
class Gaussian {
public:
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

void writeMCTree() {
    TTree* tree = new TTree("tree_mc","GUNDAM MC Sample Tree");

    Gaussian MakeA(10000);
    double varA;
    tree->Branch("A",&varA);

    Gaussian MakeB(10001);
    double varB;
    tree->Branch("B",&varB);

    Gaussian MakeC(10002);
    double varC;
    tree->Branch("C",&varC);

    for (int i=0; i<5000; ++i) {
        varA = MakeA();
        varB = MakeB();
        varC = MakeC();
        if (varC <= 0.0) varC = - varC + 1E-10;
        varA += 1.0;
        tree->Fill();
    }

    for (int i=0; i<5000; ++i) {
        varA = MakeA();
        varB = MakeB();
        varC = MakeC();
        if (varC > 0.0) varC = - varC - 1E-10;
        varA -= 1.0;
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

    for (int i=0; i<4000; ++i) {
        varA = MakeA();
        varB = MakeB();
        varA += 1.0;
        tree->Fill();
    }

    for (int i=0; i<3000; ++i) {
        varA = MakeA();
        varB = MakeB();
        varA -= 1.0;
        tree->Fill();
    }
}

int main() {
    std::shared_ptr<TFile> file(new TFile("100NormalizationTree.root","new"));

    writeMCTree();
    writeDataTree();

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
