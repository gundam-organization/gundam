#ifndef __XsecFitter_hh__
#define __XsecFitter_hh__

#include <algorithm>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <sstream>
#include <string>

#include <TGraph.h>
#include <TFile.h>
#include <TMatrixDSymEigen.h>
#include <TVirtualFitter.h>
#include <TObject.h>
#include <TMath.h>
#include <TRandom3.h>

#include "AnaSample.hh"
#include "AnySample.hh"
#include "AnaFitParameters.hh"

using namespace std;
class XsecFitter : public TObject
{
    public:
        XsecFitter(const int seed, const int num_threads);
        ~XsecFitter();
        void SetSeed(int seed);
        void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag);
        void InitFitter(std::vector<AnaFitParameters*> &fitpara, double reg, double reg2, int nipsbinsin, const std::string& paramVectorFname);
        void FixParameter(const std::string& par_name, const double& value);
        void Fit(std::vector<AnaSample*> &samples, const std::vector<std::string>& topology, int datatype, int fitMethod, int statFluct);
        void SetSaveMode(TDirectory *dirout, int freq)
        { m_dir = dirout; m_freq = freq; }
        void SetPOTRatio(double val){ m_potratio = val; }

        ClassDef(XsecFitter, 0);


        TTree *outtree;

        // Declaration of leaf types
        Int_t          cutBranch;
        Float_t        D1true;
        Float_t        D2true;
        Int_t          mectopology;
        Int_t          reaction;
        Float_t        D1Reco;
        Float_t        D2Reco;
        Float_t        weight;
        Float_t        weightNom;
        Float_t        weightMC;

        // New kinematic variables always included for phase space cuts
        Float_t        pMomRec;
        Float_t        pMomTrue;
        Float_t        muMomRec;
        Float_t        muMomTrue;
        Float_t        muCosThetaRec;
        Float_t        muCosThetaTrue;
        Float_t        pCosThetaRec;
        Float_t        pCosThetaTrue;

        void InitOutputTree()
        {
            // Set branches
            outtree->Branch("reaction", &reaction, "reaction/I");
            outtree->Branch("cutBranch", &cutBranch, "cutBranch/I");
            outtree->Branch("mectopology", &mectopology, "mectopology/I");
            outtree->Branch("D1True", &D1true, ("D1True/F"));
            outtree->Branch("D1Rec", &D1Reco, ("D1Rec/F"));
            outtree->Branch("D2True", &D2true, ("D2True/F"));
            outtree->Branch("D2Rec", &D2Reco, ("D2Rec/F"));
            outtree->Branch("weight", &weight, "weight/F");
            outtree->Branch("weightNom", &weightNom, "weightNom/F");
            outtree->Branch("weightMC", &weightMC, "weightMC/F");

            outtree->Branch("muMomRec", &muMomRec, ("muMomRec/F"));
            outtree->Branch("muMomTrue", &muMomTrue, ("muMomTrue/F"));
            outtree->Branch("muCosThetaRec", &muCosThetaRec, ("muCosThetaRec/F"));
            outtree->Branch("muCosThetaTrue", &muCosThetaTrue, ("muCosThetaTrue/F"));
            outtree->Branch("pMomRec", &pMomRec, ("pMomRec/F"));
            outtree->Branch("pMomTrue", &pMomTrue, ("pMomTrue/F"));
            outtree->Branch("pCosThetaRec", &pCosThetaRec, ("pCosThetaRec/F"));
            outtree->Branch("pCosThetaTrue", &pCosThetaTrue, ("pCosThetaTrue/F"));
        }

    private:
        void GenerateToyData(int toyindx = 0, int toytype=0, int statFluct=0);
        double FillSamples(std::vector< std::vector<double> > new_pars,
                int datatype = 0);
        void DoSaveParams(std::vector< std::vector<double> > new_pars);
        void DoSaveEvents(int fititer);
        void DoSaveFinalEvents(int fititer, std::vector< std::vector<double> > parresults);
        void DoSaveChi2();
        void CollectSampleHistos();
        void DoSaveResults(std::vector< std::vector<double> >& parresults,
                std::vector< std::vector<double> >& parerrors,
                std::vector< std::vector<double> >& parerrorsplus,
                std::vector< std::vector<double> >& parerrorsminus,
                std::vector< std::vector<double> >& parerrorspara,
                std::vector< std::vector<double> >& parerrorsglobc,
                std::vector< std::vector<double> >& parerrorsprof,
                double chi2);
        double FindProfiledError(int param, TMatrixDSym mat);


        TH1D* mcHisto;
        TH1D* mcSigHisto;
        TH1D* prefitParams;
        TRandom3* rng;
        TDirectory *m_dir;
        std::vector<AnaFitParameters*> m_fitpara;
        std::vector<int> m_nparclass;
        std::vector<AnaSample*> m_samples;
        Double_t m_potratio;
        int m_npar, m_calls, m_freq;
        Int_t nipsbins;
        Double_t reg_p1, reg_p2;
        std::string paramVectorFileName;
        std::vector<std::string> par_names;
        std::vector<double> par_prefit;
        std::vector<double> par_postfit;
        std::vector<double> par_errfit;
        std::vector<double> par_toydata;
        std::vector<double> vec_chi2_stat;
        std::vector<double> vec_chi2_sys;
        std::vector<double> vec_chi2_reg;
        std::vector<double> vec_chi2_reg2;
        int m_threads;
};
#endif
