#ifndef __AnaSample_hh__
#define __AnaSample_hh__

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <TDirectory.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TRandom3.h>

#include "AnaEvent.hh"

///////////////////////////////////////
// Class definition
///////////////////////////////////////
class AnaSample
{
    public:
        AnaSample();
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

        TH1D* GetPredHisto(){ return m_hpred; }
        TH1D* GetDataHisto(){ return m_hdata; }
        TH1D* GetMCHisto(){ return m_hmc; }
        TH1D* GetMCTruthHisto(){ return m_hmc_true; }
        TH1D* GetSignalHisto(){ return m_hsig; }

        //virtual functions
        virtual void SetData(TObject* data) = 0;
        virtual void FillEventHisto(int datatype) = 0;
        virtual double CalcChi2() const = 0;
        virtual void Write(TDirectory* dirout, const std::string& bsname, int fititer) = 0;
        virtual void GetSampleBreakdown(TDirectory *dirout, const std::string& tag, bool save) = 0;
        virtual void GetSampleBreakdown(TDirectory *dirout, const std::string& tag, const std::vector<std::string>& topology, bool save) = 0;

    protected:
        int m_sample_id;
        double m_norm;
        std::string m_name;
        std::string m_detector;
        std::vector<AnaEvent> m_events;

        TH1D* m_hmc_true;
        TH1D* m_hmc;
        TH1D* m_hpred;
        TH1D* m_hdata;
        TH1D* m_hsig;
};

#endif
