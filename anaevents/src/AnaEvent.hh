//////////////////////////////////////////////////////////
//
//  A container with subset of necessary event
//  information for CCQE analysis
//
//
//  Created: Thu Jun  6 11:28:13 CEST 2013
//  Modified:
//
//////////////////////////////////////////////////////////
#ifndef __AnaEvent_hh__
#define __AnaEvent_hh__

#include <iostream>

#include <TMath.h>

class AnaEvent
{
    public:
        AnaEvent(long int evid) //unique event id
        {
            m_evid     = evid;
            m_topology = -1;
            m_reaction = -1;
            m_sample   = -1;
            m_signal   = false;
            m_trueEvt  = false;
            m_trueEnu  = -999.0;
            m_recEnu   = -999.0;
            m_trueD1   = -999.0;
            m_trueD2   = -999.0;
            m_recD1    = -999.0;
            m_recD2    = -999.0;
            m_wght     = 1.0;
            m_wghtMC   = 1.0;

            // New kinematic variables always included for phase space cuts
            m_pMomRec = -999.0;
            m_pMomTrue = -999.0;
            m_muMomRec = -999.0;
            m_muMomTrue = -999.0;
            m_muCosThetaRec = -999.0;
            m_muCosThetaTrue = -999.0;
            m_pCosThetaRec = -999.0;
            m_pCosThetaTrue = -999.0;
        }

        //Set/Get methods
        void SetTopology(int val){ m_topology = val; }
        int GetTopology(){ return m_topology; }

        void SetReaction(int val){ m_reaction = val; }
        int GetReaction(){ return m_reaction; }

        void SetSampleType(int val){ m_sample = val; }
        int GetSampleType(){ return m_sample; }

        void SetSignalEvent(const bool flag = true){ m_signal = flag; }
        bool isSignalEvent(){ return m_signal; }

        void SetTrueEvent(const bool flag = true){ m_trueEvt = flag; }
        bool isTrueEvent(){ return m_trueEvt; }

        long int GetEvId(){ return m_evid; }

        void SetTrueEnu(double val) {m_trueEnu = val;}
        double GetTrueEnu(){ return m_trueEnu; }

        void SetRecEnu(double val){ m_recEnu = val; }
        double GetRecEnu(){ return m_recEnu; }

        void SetTrueD1(double val){ m_trueD1 = val; }
        double GetTrueD1(){ return m_trueD1; }

        void SetRecD1(double val){ m_recD1 = val; }
        double GetRecD1(){ return m_recD1; }

        void SetTrueD2(double val){ m_trueD2 = val; }
        double GetTrueD2(){ return m_trueD2; }

        void SetRecD2(double val){ m_recD2 = val; }
        double GetRecD2(){ return m_recD2; }

        void SetEvWght(double val){ m_wght  = val; }
        void SetEvWghtMC(double val){ m_wghtMC  = val; }
        void AddEvWght(double val){ m_wght *= val; }
        double GetEvWght(){ return m_wght; }
        double GetEvWghtMC(){ return m_wghtMC; }

        void ResetEvWght(){ m_wght = m_wghtMC; }
        // New kinematic variables always included for phase space cuts

        void SetQ2(double val){m_q2 = val;}
        double GetQ2() const { return m_q2; }

        void SetpMomRec(double val){ m_pMomRec = val; }
        void SetpMomTrue(double val){ m_pMomTrue = val; }
        void SetmuMomRec(double val){ m_muMomRec = val; }
        void SetmuMomTrue(double val){ m_muMomTrue = val; }
        void SetmuCosThetaRec(double val){ m_muCosThetaRec = val; }
        void SetmuCosThetaTrue(double val){ m_muCosThetaTrue = val; }
        void SetpCosThetaRec(double val){ m_pCosThetaRec = val; }
        void SetpCosThetaTrue(double val){ m_pCosThetaTrue = val; }

        double GetpMomRec(){ return m_pMomRec; }
        double GetpMomTrue(){ return m_pMomTrue; }
        double GetmuMomRec(){ return m_muMomRec; }
        double GetmuMomTrue(){ return m_muMomTrue; }
        double GetmuCosThetaRec(){ return m_muCosThetaRec; }
        double GetmuCosThetaTrue(){ return m_muCosThetaTrue; }
        double GetpCosThetaRec(){ return m_pCosThetaRec; }
        double GetpCosThetaTrue(){ return m_pCosThetaTrue; }

        void Print()
        {
            std::cout << "Event ID        " << m_evid << std::endl
                      << "Topology        " << GetTopology() << std::endl
                      << "Sample          " << GetSampleType() << std::endl
                      << "True energy     " << GetTrueEnu() << std::endl
                      << "Recon energy    " << GetRecEnu() << std::endl
                      << "True D1         " << GetTrueD1() << std::endl
                      << "Reco D1         " << GetRecD1() << std::endl
                      << "True D2         " << GetTrueD2() << std::endl
                      << "Reco D2         " << GetRecD2() << std::endl
                      << "Event weight    " << GetEvWght() << std::endl
                      << "Event weight MC " << GetEvWghtMC() << std::endl
                      << "Reco proton momentum  " << GetpMomRec() << std::endl
                      << "True proton momentum  " << GetpMomTrue() << std::endl
                      << "Reco muon momentum    " << GetmuMomRec() << std::endl
                      << "True muon momentum    " << GetmuMomTrue() << std::endl
                      << "Reco muon cos theta   " << GetmuCosThetaRec() << std::endl
                      << "True muon cos theta   " << GetmuCosThetaTrue() << std::endl
                      << "Reco Proton Cos Theta " << GetpCosThetaRec() << std::endl
                      << "True Proton Cos Theta " << GetpCosThetaTrue() << std::endl;
        }

    private:
        long int m_evid;   //unique event id
        int m_topology;    //final state topology type
        int m_reaction;    //event interaction mode
        int m_sample;      //sample type (aka cutBranch)
        bool m_signal;     //flag if signal event
        bool m_trueEvt;    //flag if true event
        double m_trueEnu;  //true nu energy
        double m_recEnu;   //recon nu energy
        double m_trueD1;   //true D1
        double m_trueD2;   //true D2
        double m_recD1;    //reco D1
        double m_recD2;    //reco D2
        double m_q2;
        double m_wght;     //event weight
        double m_wghtMC;   //event weight from original MC

        // New kinematic variables always included for phase space cuts
        double m_muMomRec;
        double m_muMomTrue;
        double m_muCosThetaRec;
        double m_muCosThetaTrue;
        double m_pMomRec;
        double m_pMomTrue;
        double m_pCosThetaRec;
        double m_pCosThetaTrue;
};

#endif
