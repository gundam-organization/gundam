//////////////////////////////////////////////////////////
//
//  A class for event samples for for CCQE analysis
//
//
//
//  Created: Thu Jun  6 12:01:10 CEST 2013
//  Modified:
//
//////////////////////////////////////////////////////////
#ifndef __AnaSample_hh__
#define __AnaSample_hh__

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <TH2D.h>
#include <TDirectory.h>
#include <TRandom3.h>

#include "AnaEvent.hh"
using namespace std;

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
        int GetSampleType(){ return m_sampleid; }
        AnaEvent* GetEvent(int evnum);
        void AddEvent(AnaEvent& event);
        void ResetWeights();
        void PrintStats();
        void SetNorm(double val){ m_norm = val; }
        double GetNorm(){ return m_norm; }
        bool isIngrid(){ return m_ingrid; }

        //virtual functions
        virtual void SetData(TObject *data) = 0;
        virtual void FillEventHisto(int datatype) = 0;
        virtual double CalcChi2() = 0;
        virtual void Write(TDirectory *dirout, const char *bsname, int fititer) = 0;
        virtual void GetSampleBreakdown(TDirectory *dirout, const std::string& tag, bool save) = 0;
        virtual void GetSampleBreakdown(TDirectory *dirout, const std::string& tag, const std::vector<std::string>& topology, bool save) = 0;

    protected:
        int m_sampleid;
        bool m_ingrid; //is this an ingrid or an ND280 sample?
        double m_norm;
        std::string m_name;
        std::vector<AnaEvent> m_events;
};

#endif
