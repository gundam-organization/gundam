#ifndef __AnaSample_hh__
#define __AnaSample_hh__

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <TDirectory.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TRandom3.h>
#include <TTree.h>

#include "AnaEvent.hh"

///////////////////////////////////////
// Class definition
///////////////////////////////////////
class AnaSample
{
    public:
        AnaSample(int sample_id, const std::string& name, const std::string& detector,
                std::vector<std::pair <double,double> > v_d1edges,
                std::vector<std::pair <double,double> > v_d2edges, TTree *data, bool isBuffer, bool useSample=true);
        ~AnaSample();

        void ClearEvents();
        int GetN();
        AnaEvent* GetEvent(int evnum);
        void AddEvent(AnaEvent& event);
        void ResetWeights();
        void PrintStats();
        void SetNorm(double val){ m_norm = val; }
        int GetSampleID(){ return m_sample_id; }
        double GetNorm(){ return m_norm; }
        std::string GetName(){ return m_name; }
        std::string GetDetector(){ return m_detector; }

        void SetD1Binning(int nbins, double *bins);
        void SetD2Binning(int nbins, double *bins);
        void SetEnuBinning(int nbins, double *bins);
        int GetAnyBinIndex(const double D1, const double D2) const;
        void MakeHistos(); //must be called after binning is changed

        TH1D* GetPredHisto(){ return m_hpred; }
        TH1D* GetDataHisto(){ return m_hdata; }
        TH1D* GetMCHisto(){ return m_hmc; }
        TH1D* GetMCTruthHisto(){ return m_hmc_true; }
        TH1D* GetSignalHisto(){ return m_hsig; }

        //virtual functions
        void SetData(TObject* data);
        void FillEventHisto(int datatype);
        double CalcChi2() const;
        void Write(TDirectory* dirout, const std::string& bsname, int fititer);
        void GetSampleBreakdown(TDirectory *dirout, const std::string& tag, const std::vector<std::string>& topology, bool save);

    protected:
        int m_sample_id;
        double m_norm;
        bool m_use_sample; // If false, we won't include any events in this sample (useful for testing the effect of removing samples)
        bool m_BufferBin; // Should we bother plotting the last bin (dat, dphit), or is it just a buffer (dpt)?
        std::string m_name;
        std::string m_detector;
        std::vector<AnaEvent> m_events;
        std::vector<std::pair<double, double> > m_D1edges;
        std::vector<std::pair<double, double> > m_D2edges;

        TTree* m_data_tree;
        TH1D* m_hmc_true;
        TH1D* m_hmc;
        TH1D* m_hpred;
        TH1D* m_hdata;
        TH1D* m_hsig;

        int nbins_D1, nbins_D2, nbins_enu, nAnybins, nbinsD1_toPlot;
        double *bins_D1, *bins_D2, *bins_enu, *bins_Any, *bins_D1toPlot;
};

#endif
