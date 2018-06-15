//This is the code that actually reads int he MC tree and fills the event info.
//The tree should be produced by feeding a HL2 microtree into the treeconvert macro.

#include "AnyTreeMC.hh"

AnyTreeMC::AnyTreeMC(const std::string& file_name, const std::string& tree_name)
{
    fChain = new TChain(tree_name.c_str());
    fChain -> Add(file_name.c_str());
    SetBranches();
}

AnyTreeMC::~AnyTreeMC()
{
    if(fChain == nullptr)
        return;
    delete fChain -> GetCurrentFile();
}

long int AnyTreeMC::GetEntry(long int entry) const
{
    // Read contents of entry.
    if(fChain == nullptr)
        return -1;
    else
        return fChain -> GetEntry(entry);
}

void AnyTreeMC::SetBranches()
{
    // Set branch addresses and branch pointers
    fChain -> SetBranchAddress("cutBranch", &cutBranch, &b_cutBranch);
    fChain -> SetBranchAddress("mectopology", &evtTopology, &b_evtTopology);
    fChain -> SetBranchAddress("reaction", &evtReaction, &b_evtReaction);
    fChain -> SetBranchAddress("D1True", &D1True, &b_D1True);
    fChain -> SetBranchAddress("D2True", &D2True, &b_D2True);
    fChain -> SetBranchAddress("D1Rec", &D1Reco, &b_D1Reco);
    fChain -> SetBranchAddress("D2Rec", &D2Reco, &b_D2Reco);
    fChain -> SetBranchAddress("Enureco", &EnuReco, &b_EnuReco);
    fChain -> SetBranchAddress("Enutrue", &EnuTrue, &b_EnuTrue);
    fChain -> SetBranchAddress("weight", &weight, &b_weight);

    // New kinematic variables always included for phase space cuts
    fChain -> SetBranchAddress("pMomRec", &pMomRec, &b_pMomRec);
    fChain -> SetBranchAddress("pMomTrue", &pMomTrue, &b_pMomTrue);
    fChain -> SetBranchAddress("pCosThetaRec", &pCosThetaRec, &b_pCosThetaRec);
    fChain -> SetBranchAddress("pCosThetaTrue", &pCosThetaTrue, &b_pCosThetaTrue);
    fChain -> SetBranchAddress("muMomRec", &muMomRec, &b_muMomRec);
    fChain -> SetBranchAddress("muMomTrue", &muMomTrue, &b_muMomTrue);
    fChain -> SetBranchAddress("muCosThetaRec", &muCosThetaRec, &b_muCosThetaRec);
    fChain -> SetBranchAddress("muCosThetaTrue", &muCosThetaTrue, &b_muCosThetaTrue);
}

void AnyTreeMC::GetEvents(std::vector<AnaSample*>& ana_samples, const std::vector<int>& sig_topology)
{
    if(fChain == nullptr) return;
    if(ana_samples.empty()) return;

    long int nentries = fChain -> GetEntries();
    long int nbytes = 0;

    std::cout << "[AnyTreeMC]: Reading events...\n";
    for(long int jentry = 0; jentry < nentries; jentry++)
    {
        if(jentry % (int)1e+5 == 0)
            std::cout << "[AnyTreeMC]: Processing event " << jentry << " out of " << nentries << std::endl;
        nbytes += fChain -> GetEntry(jentry);
        //create and fill event structure
        AnaEvent ev(jentry);
        ev.SetSampleType(cutBranch);
        ev.SetTopology(evtTopology); // mectopology (i.e. CC0Pi,CC1Pi etc)
        ev.SetReaction(evtReaction); // reaction (i.e. CCQE,CCRES etc)
        ev.SetTrueEnu(EnuTrue/1000.0);   //MeV - ->  GeV
        ev.SetRecEnu(EnuReco/1000.0); //MeV - ->  GeV
        ev.SetTrueD1(D1True);
        ev.SetRecD1(D1Reco);
        ev.SetTrueD2(D2True);
        ev.SetRecD2(D2Reco);
        ev.SetEvWght(weight);
        ev.SetEvWghtMC(weight);

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
