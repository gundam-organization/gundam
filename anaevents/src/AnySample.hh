//////////////////////////////////////////////////////////
//
//  A class for event samples for for any analysis
//
//
//
//  Created: Nov 17 2015
//
//////////////////////////////////////////////////////////
#ifndef __AnySample_hh__
#define __AnySample_hh__

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <TDirectory.h>
#include <TH2D.h>
#include <TRandom3.h>
#include <TTree.h>

#include "AnaEvent.hh"
#include "AnaSample.hh"

///////////////////////////////////////
// Class definition
///////////////////////////////////////
class AnySample : public AnaSample
{
    public:
        AnySample(int sample_id, const std::string& name, const std::string& detector,
                std::vector<std::pair <double,double> > v_d1edges,
                std::vector<std::pair <double,double> > v_d2edges, TTree *data, bool isBuffer, bool useSample=true);
        ~AnySample();

        //binning for various histograms
        void SetD1Binning(int nbins, double *bins);
        void SetD2Binning(int nbins, double *bins);
        void SetEnuBinning(int nbins, double *bins);
        int GetAnyBinIndex(const double D1, const double D2);
        void MakeHistos(); //must be called after binning is changed

        //histogram for event distributions
        void SetData(TObject *hdata);
        void FillEventHisto(int datatype);
        double CalcChi2();

        TH1D* GetPredHisto(){ return m_hpred; }
        TH1D* GetDataHisto(){ return m_hdata; }
        TH1D* GetMCHisto(){ return m_hmc; }
        TH1D* GetMCTruthHisto(){ return m_hmc_true; }
        TH1D* GetSignalHisto(){ return m_hsig; }

        void GetSampleBreakdown(TDirectory *dirout, const std::string& tag, bool save);
        void GetSampleBreakdown(TDirectory *dirout, const std::string& tag, const std::vector<std::string>& topology, bool save);
        void Write(TDirectory *dirout, const std::string& bsname, int fititer);
        int GetSampleId(){ return sample_id; }

    private:
        int sample_id;
        TH1D *m_hmc_true;
        TH1D *m_hmc;
        TH1D *m_hpred;
        TH1D *m_hdata;
        TH1D *m_hsig;
        TTree * m_data_tree;
        int nbins_D1, nbins_D2, nbins_enu, nAnybins, nbinsD1_toPlot;
        double *bins_D1, *bins_D2, *bins_enu, *bins_Any, *bins_D1toPlot;
        std::vector<std::pair<double, double> > m_D1edges;
        std::vector<std::pair<double, double> > m_D2edges;
        bool m_use_sample; // If true, we won't include any events in this sample (useful for testing the effect of removing samples)
        bool m_BufferBin; // Should we bother plotting the last bin (dat, dphit), or is it just a buffer (dpt)?
};

#endif
