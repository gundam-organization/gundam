//This is the code that actually reads int he MC tree and fills the event info.
//The tree should be produced by feeding a HL2 microtree into the treeconvert macro.

#include "AnaTreeMC.hh"

AnaTreeMC::AnaTreeMC(const std::string& file_name, const std::string& tree_name)
{
    fChain = new TChain(tree_name.c_str());
    fChain -> Add(file_name.c_str());
    SetBranches();
}

AnaTreeMC::~AnaTreeMC()
{
    if(fChain == nullptr)
        return;
    delete fChain -> GetCurrentFile();
}

long int AnaTreeMC::GetEntry(long int entry) const
{
    // Read contents of entry.
    if(fChain == nullptr)
        return -1;
    else
        return fChain -> GetEntry(entry);
}

void AnaTreeMC::SetBranches()
{
    // Set branch addresses and branch pointers
    fChain -> SetBranchAddress("nutype", &nutype);
    fChain -> SetBranchAddress("cut_branch", &cutBranch);
    fChain -> SetBranchAddress("topology", &evtTopology);
    fChain -> SetBranchAddress("reaction", &evtReaction);
    fChain -> SetBranchAddress("D1True", &D1True);
    fChain -> SetBranchAddress("D1Reco", &D1Reco);
    fChain -> SetBranchAddress("D2True", &D2True);
    fChain -> SetBranchAddress("D2Reco", &D2Reco);
    fChain -> SetBranchAddress("q2_true", &Q2True);
    fChain -> SetBranchAddress("q2_reco", &Q2Reco);
    fChain -> SetBranchAddress("enu_true", &EnuTrue);
    fChain -> SetBranchAddress("enu_reco", &EnuReco);
    fChain -> SetBranchAddress("weight", &weight);

    // New kinematic variables always included for phase space cuts
    fChain -> SetBranchAddress("pMomRec", &pMomRec);
    fChain -> SetBranchAddress("pMomTrue", &pMomTrue);
    fChain -> SetBranchAddress("pCosThetaRec", &pCosThetaRec);
    fChain -> SetBranchAddress("pCosThetaTrue", &pCosThetaTrue);
    fChain -> SetBranchAddress("muMomRec", &muMomRec);
    fChain -> SetBranchAddress("muMomTrue", &muMomTrue);
    fChain -> SetBranchAddress("muCosThetaRec", &muCosThetaRec);
    fChain -> SetBranchAddress("muCosThetaTrue", &muCosThetaTrue);
}

void AnaTreeMC::GetEvents(std::vector<AnaSample*>& ana_samples, const std::vector<int>& sig_topology, const bool evt_type)
{
    if(fChain == nullptr || ana_samples.empty()) return;

    long int nentries = fChain -> GetEntries();
    long int nbytes = 0;

    std::cout << "[AnaTreeMC]: Reading events...\n";
    for(long int jentry = 0; jentry < nentries; jentry++)
    {
        if(jentry % static_cast<long int>(1E5) == 0)
            std::cout << "[AnaTreeMC]: Processing event " << jentry << " out of " << nentries << std::endl;
        nbytes += fChain -> GetEntry(jentry);
        //create and fill event structure
        AnaEvent ev(jentry);
        ev.SetTrueEvent(evt_type);
        ev.SetFlavor(nutype);
        ev.SetSampleType(cutBranch);
        ev.SetTopology(evtTopology); // mectopology (i.e. CC0Pi,CC1Pi etc)
        ev.SetReaction(evtReaction); // reaction (i.e. CCQE,CCRES etc)
        ev.SetTrueEnu(EnuTrue);
        ev.SetRecoEnu(EnuReco);
        ev.SetTrueD1(D1True);
        ev.SetRecD1(D1Reco);
        ev.SetTrueD2(D2True);
        ev.SetRecD2(D2Reco);
        ev.SetEvWght(weight);
        ev.SetEvWghtMC(weight);
        ev.SetQ2True(Q2True);
        ev.SetQ2Reco(Q2Reco);

        ev.SetmuMomRec(muMomRec);
        ev.SetmuMomTrue(muMomTrue);
        ev.SetmuCosThetaRec(muCosThetaRec);
        ev.SetmuCosThetaTrue(muCosThetaTrue);
        ev.SetpMomRec(pMomRec);
        ev.SetpMomTrue(pMomTrue);
        ev.SetpCosThetaRec(pCosThetaRec);
        ev.SetpCosThetaTrue(pCosThetaTrue);

        for(const auto& signal_topology : sig_topology)
        {
            if(signal_topology == evtTopology)
            {
                ev.SetSignalEvent();
                break;
            }
        }

        for(auto& sample : ana_samples)
        {
            if(sample -> GetSampleID() == cutBranch)
                sample -> AddEvent(ev);
        }
    }

    for(auto& sample : ana_samples)
        sample -> PrintStats();
}
