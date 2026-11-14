"""Small deterministic inputs for the finite-difference gradient tests."""
from pathlib import Path
import subprocess


def writeNormalization():
    import numpy as np
    import uproot
    centers = np.array([-3., -1.5, -.5, .5, 1.5, 3.])
    a = np.repeat(centers, 6)
    b = np.tile(centers, 6)
    with uproot.recreate('100NormalizationTree.root') as output:
        mc = output.mktree('tree_mc', {'A': 'float64', 'B': 'float64', 'C': 'float64'})
        mc.extend({'A': np.tile(a, 4), 'B': np.tile(b, 4),
                   'C': np.repeat([1., 1., -1., -1.], len(a))})
        data = output.mktree('tree_dt', {'A': 'float64', 'B': 'float64'})
        data.extend({'A': a, 'B': b})


def writeCovariance():
    # ROOT object branches and covariance objects need a small native fixture.
    macro = Path.cwd() / '580NumericsFitGradientSpline-input.C'
    macro.write_text(r'''#include <TFile.h>
#include <TTree.h>
#include <TGraph.h>
#include <TMatrixDSym.h>
#include <TVectorD.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TSystem.h>
#include <cmath>
{
 {
 TFile file("100CovarianceTree.root", "RECREATE");
 double a, b, c;
 TGraph curveC(5), curveD(5);
 TGraph* pc = &curveC;
 TGraph* pd = &curveD;
 TTree mc("tree_mc", "tree_mc");
 mc.Branch("A", &a, "A/D"); mc.Branch("B", &b, "B/D"); mc.Branch("C", &c, "C/D");
 mc.Branch("spline_C", "TGraph", &pc, 32000, 0);
 mc.Branch("spline_D", "TGraph", &pd, 32000, 0);
 double centers[] = {-3., -1.5, -.5, .5, 1.5, 3.};
 for (int sign : {-1, 1}) for (double x : centers) for (double y : centers) {
   a = x; b = y; c = sign;
   for (int k = 0; k < 5; ++k) {
     double p = k - 2.;
     curveC.SetPoint(k, p, std::exp((sign > 0 ? .2 : .05)*p));
     curveD.SetPoint(k, p, std::exp((sign < 0 ? .3 : .05)*p));
   }
   mc.Fill();
 }
 mc.Write();
 TTree data("tree_dt", "tree_dt");
 data.Branch("A", &a, "A/D"); data.Branch("B", &b, "B/D");
 for (double x : centers) for (double y : centers) { a = x; b = y; data.Fill(); }
 data.Write();
 TMatrixDSym cov(4);
 for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
   cov(i,j) = i == j ? .09 : .045 * ((i+j)%2 ? -1 : 1);
 cov.Write("CovarianceInputCovariance");
 TVectorD priors(4); priors[0] = priors[1] = 1.; priors[2] = priors[3] = 0.;
 priors.Write("CovarianceInputPriors");
 TObjArray names; names.SetOwner(true);
 for (const char* name : {"norm_A", "norm_B", "spline_C", "spline_D"}) names.Add(new TObjString(name));
 names.Write("CovarianceInputNames", 1);
 }
 gSystem->Exit(0);
}
''')
    subprocess.run(['root', '-n', '-l', '-b', '-q', str(macro)], check=True)
