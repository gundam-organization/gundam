# !/bin/bash
# Wrap a ROOT macro as a script.
root <<EOF

#include <iostream>
#include <sstream>
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
//    (At, Bt) -- These are the true values for the event.
//    spline_CD -- An event-by-event weighting spline (2D)
//
//    Variable "A" is generated as a normal distribution centered at At.
//    Variable "B" is generated as a normal distribution centered at At.
//    Variable "At" is generated as a normal distribution centered at 0.0.
//    Variable "Bt" is generated as a normal distribution centered at 0.0.

// tree_dt variables:
//
//    (A, B) -- Analogs to the detector reconstructed values.
//
// A covariance matrix is defined for the fit parameters.  The fit parameters
// are:
//
//    norm_A -- The normalization.
//    var_C -- An event weight spline for C greater than zero.
//    var_D -- An event weight spline for C less than zero.

const double trueC = 0.0;
const double trueD = 0.0;
const double resA = 0.1;
const double resB = 0.1;

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
class WeightingHistogram {
public:
    /// Create a generator with a particular exponential scale factor.
    explicit WeightingHistogram(double origC, double origD) {
        _origC_ = origC;
        _origD_ = origD;
    }

    double shiftC(double c) {return 0.01*c;}
    double shiftD(double d) {return 0.01*d;}

    double func(double x, double y, double c, double d) {
        double scale = 0.1;
        double xx = x + shiftC(c);
        double yy = y + shiftD(d);
        return std::exp(-xx*xx/2.0)*std::exp(-yy*yy/2.0);
    }

    double weight(double x, double y, double c, double d) {
        double f1 = func(x,y,c,d);
        double f2 = func(x,y,_origC_, _origD_);
        return f2/f1;
    }

    /// Fill the graph.
    void operator()(TObject* obj, double x, double y) {
        TH2D* hist = dynamic_cast<TH2D*>(obj);
        int n = hist->GetXaxis()->GetNbins();
        int m = hist->GetYaxis()->GetNbins();
        for (int i=1; i<=n; ++i) {
            for (int j=1; j<=m; ++j) {
                double c = hist->GetXaxis()->GetBinCenter(i);
                double d = hist->GetYaxis()->GetBinCenter(j);
                int bin = hist->GetBin(i,j);
                double w = weight(x, y, c, d);
                hist->SetBinContent(bin,w);
            }
        }
    }

private:
    double _origC_{std::nan("unset")};
    double _origD_{std::nan("unset")};
};

void writeMCTree() {
    TTree* tree = new TTree("tree_mc","GUNDAM MC Sample Tree with splines");
    TH2D* priorHist
         = new TH2D("priorHist", "Prior MC", 50, -5.0, 5.0, 50, -5.0, 5.0);
    TH2D* weightedHist
         = new TH2D("weightedHist", "Weighted MC", 50, -5.0, 5.0, 50, -5.0, 5.0);

    double varA;
    tree->Branch("A",&varA);

    double varB;
    tree->Branch("B",&varB);

    Gaussian MakeAt(10000);
    double varAt;
    tree->Branch("At",&varAt);

    Gaussian MakeBt(10001);
    double varBt;
    tree->Branch("Bt",&varBt);

    WeightingHistogram fillCD(0.0, 0.0);
    TClonesArray spline_CD("TH2D",1);
    new(spline_CD[0]) TH2D("splineCD", "A weight histogram",
                           7, -3.5, 3.5,
                           7, -3.5, 3.5);
    tree->Branch("spline_CD",&spline_CD,32000,0);

    // These events are reweighted by
    // spline_C in the fit, and spline_D is set to be flat.
    std::string name("100MultParTreeSplines.pdf");
    std::string suffix("(");
    int stride = 1000;
    int i = 0;
    while (i < 100000) {
        std::ostringstream title;
        do {varAt = MakeAt();} while (std::abs(varAt) > 5.0);
        do {varA = gRandom->Gaus(varAt,resA);} while (std::abs(varA) > 5.0);
        do {varBt = MakeBt();} while (std::abs(varBt) > 5.0);
        do {varB = gRandom->Gaus(varBt,resB);} while (std::abs(varB) > 5.0);
        title << "Weight(C,D) at"
              << " At " << varAt << " (" << varA << ")"
              << " Bt " << varBt << " (" << varB << ")";
        fillCD(spline_CD[0],varAt,varBt);
        ((TH2*)spline_CD[0])->SetTitle(title.str().c_str());
        if (i % stride == 0) {
            ((TH2*)spline_CD[0])->SetStats(false);
            spline_CD[0]->Draw("colz,cont3");
            gPad->Print((name+suffix).c_str());
            suffix = "";
        }
        tree->Fill();
        priorHist->Fill(varA, varB);
        double w = fillCD.weight(varAt, varBt, trueC, trueD);
        weightedHist->Fill(varA, varB, w);
        ++i;
    }
    spline_CD[0]->Draw("colz,cont3");
    gPad->Print((name + ")").c_str());

}

void writeDataTree() {
    TH2D* dataHist
        = new TH2D("dataHist", "Fake Data", 50, -5.0, 5.0, 50, -5.0, 5.0);
    TTree* tree = new TTree("tree_dt","GUNDAM Data Sample Tree");

    double varA;
    Gaussian MakeAt(20000);
    double varAt;
    tree->Branch("A",&varA);

    double varB;
    Gaussian MakeBt(20001);
    double varBt;
    tree->Branch("B",&varB);

    double maxW = 100.0;

    WeightingHistogram fillCD(0.0, 0.0);

    // Write a data sample.  These are biased.
    int i = 0;
    while (i < 10000) {
        do {varAt = MakeAt() + fillCD.shiftC(trueC);} while (std::abs(varAt) > 5.0);
        do {varA = gRandom->Gaus(varAt,resA);} while (std::abs(varA) > 5.0);
        do {varBt = MakeBt() + fillCD.shiftD(trueD);} while (std::abs(varBt) > 5.0);
        do {varB = gRandom->Gaus(varBt,resB);} while (std::abs(varB) > 5.0);
        tree->Fill();
        dataHist->Fill(varA, varB);
        ++i;
    }
}

int main() {
    std::shared_ptr<TFile> file(new TFile("100MultiParTree.root","recreate"));
    if (!file || !file->IsOpen()) {
        std::cout << "FAIL: TFile not opened" << std::endl;
        return 1;
    }

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
