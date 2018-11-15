#ifndef __AnaEvent_hh__
#define __AnaEvent_hh__

#include <iostream>

#include <TMath.h>

#include <FitStructs.hh>

class AnaEvent
{
    public:
        AnaEvent(long int evid)
        {
            m_evid     = evid;
            m_flavor   = -1;
            m_topology = -1;
            m_reaction = -1;
            m_target   = -1;
            m_sample   = -1;
            m_signal   = false;
            m_true_evt = false;
            m_enu_true = -999.0;
            m_enu_reco = -999.0;
            m_trueD1   = -999.0;
            m_trueD2   = -999.0;
            m_recoD1   = -999.0;
            m_recoD2   = -999.0;
            m_q2_true  = -999.0;
            m_q2_reco  = -999.0;
            m_wght     = 1.0;
            m_wghtMC   = 1.0;
        }

        //Set/Get methods
        void SetTopology(int val){ m_topology = val; }
        int GetTopology(){ return m_topology; }

        void SetReaction(int val){ m_reaction = val; }
        int GetReaction(){ return m_reaction; }

        void SetTarget(int val){ m_target = val; }
        int GetTarget(){ return m_target; }

        void SetSampleType(int val){ m_sample = val; }
        int GetSampleType(){ return m_sample; }

        void SetSignalEvent(const bool flag = true){ m_signal = flag; }
        bool isSignalEvent(){ return m_signal; }

        void SetTrueEvent(const bool flag = true){ m_true_evt = flag; }
        bool isTrueEvent(){ return m_true_evt; }

        void SetFlavor(const int flavor){ m_flavor = flavor; }
        int GetFlavor(){ return m_flavor; }

        long int GetEvId(){ return m_evid; }

        void SetTrueEnu(double val) {m_enu_true = val;}
        double GetTrueEnu(){ return m_enu_true; }

        void SetRecoEnu(double val){ m_enu_reco = val; }
        double GetRecoEnu(){ return m_enu_reco; }

        void SetTrueD1(double val){ m_trueD1 = val; }
        double GetTrueD1(){ return m_trueD1; }

        void SetRecoD1(double val){ m_recoD1 = val; }
        double GetRecoD1(){ return m_recoD1; }

        void SetTrueD2(double val){ m_trueD2 = val; }
        double GetTrueD2(){ return m_trueD2; }

        void SetRecoD2(double val){ m_recoD2 = val; }
        double GetRecoD2(){ return m_recoD2; }

        void SetEvWght(double val){ m_wght  = val; }
        void SetEvWghtMC(double val){ m_wghtMC  = val; }
        void AddEvWght(double val){ m_wght *= val; }
        double GetEvWght(){ return m_wght; }
        double GetEvWghtMC(){ return m_wghtMC; }

        void ResetEvWght(){ m_wght = m_wghtMC; }

        void SetQ2Reco(double val){m_q2_reco = val;}
        double GetQ2Reco() const { return m_q2_reco; }

        void SetQ2True(double val){m_q2_true = val;}
        double GetQ2True() const { return m_q2_true; }

        void Print()
        {
            std::cout << "Event ID    " << m_evid << std::endl
                      << "Topology    " << GetTopology() << std::endl
                      << "Reaction    " << GetReaction() << std::endl
                      << "Target      " << GetTarget() << std::endl
                      << "Flavor      " << GetFlavor() << std::endl
                      << "Sample      " << GetSampleType() << std::endl
                      << "True energy " << GetTrueEnu() << std::endl
                      << "Reco energy " << GetRecoEnu() << std::endl
                      << "True D1     " << GetTrueD1() << std::endl
                      << "Reco D1     " << GetRecoD1() << std::endl
                      << "True D2     " << GetTrueD2() << std::endl
                      << "Reco D2     " << GetRecoD2() << std::endl
                      << "Weight      " << GetEvWght() << std::endl
                      << "Weight MC   " << GetEvWghtMC() << std::endl;
        }

    private:
        long int m_evid;   //unique event id
        int m_flavor;      //flavor of neutrino (numu, etc.)
        int m_topology;    //final state topology type
        int m_reaction;    //event interaction mode
        int m_target;      //target nuclei
        int m_sample;      //sample type (aka cutBranch)
        bool m_signal;     //flag if signal event
        bool m_true_evt;   //flag if true event
        double m_enu_true; //true nu energy
        double m_enu_reco; //recon nu energy
        double m_trueD1;   //true D1
        double m_trueD2;   //true D2
        double m_recoD1;   //reco D1
        double m_recoD2;   //reco D2
        double m_q2_true;
        double m_q2_reco;
        double m_wght;     //event weight
        double m_wghtMC;   //event weight from original MC
};

#endif
