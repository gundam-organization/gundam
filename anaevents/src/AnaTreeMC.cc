// This is the code that actually reads int he MC tree and fills the event info.
// The tree should be produced by feeding a HL2 microtree into the treeconvert macro.

#include "AnaTreeMC.hh"

AnaTreeMC::AnaTreeMC(const std::string& file_name, const std::string& tree_name, bool extra_var)
    : read_extra_var(extra_var)
{
    fChain = new TChain(tree_name.c_str());
    fChain->Add(file_name.c_str());
    SetBranches();
}

AnaTreeMC::~AnaTreeMC()
{
    if(fChain == nullptr)
        return;
    delete fChain->GetCurrentFile();
}

long int AnaTreeMC::GetEntry(long int entry) const
{
    // Read contents of entry.
    if(fChain == nullptr)
        return -1;
    else
        return fChain->GetEntry(entry);
}

void AnaTreeMC::SetBranches()
{
    // Set branch addresses and branch pointers
    fChain->SetBranchAddress("nutype", &nutype);
    fChain->SetBranchAddress("cut_branch", &cutBranch);
    fChain->SetBranchAddress("topology", &evtTopology);
    fChain->SetBranchAddress("reaction", &evtReaction);
    fChain->SetBranchAddress("D1True", &D1True);
    fChain->SetBranchAddress("D1Reco", &D1Reco);
    fChain->SetBranchAddress("D2True", &D2True);
    fChain->SetBranchAddress("D2Reco", &D2Reco);
    fChain->SetBranchAddress("q2_true", &Q2True);
    fChain->SetBranchAddress("q2_reco", &Q2Reco);
    fChain->SetBranchAddress("enu_true", &EnuTrue);
    fChain->SetBranchAddress("enu_reco", &EnuReco);
    fChain->SetBranchAddress("weight", &weight);

    // New kinematic variables always included for phase space cuts
    if(read_extra_var)
    {
        fChain->SetBranchAddress("pMomRec", &pMomRec);
        fChain->SetBranchAddress("pMomTrue", &pMomTrue);
        fChain->SetBranchAddress("pCosThetaRec", &pCosThetaRec);
        fChain->SetBranchAddress("pCosThetaTrue", &pCosThetaTrue);
        fChain->SetBranchAddress("muMomRec", &muMomRec);
        fChain->SetBranchAddress("muMomTrue", &muMomTrue);
        fChain->SetBranchAddress("muCosThetaRec", &muCosThetaRec);
        fChain->SetBranchAddress("muCosThetaTrue", &muCosThetaTrue);
    }
}

void AnaTreeMC::GetEvents(std::vector<AnaSample*>& ana_samples,
                          const std::vector<int>& sig_topology, const bool evt_type)
{
    if(fChain == nullptr || ana_samples.empty())
        return;

    ProgressBar pbar(60, "#");
    pbar.SetRainbow();
    pbar.SetPrefix(std::string(TAG + "Reading Events "));

    long int nentries = fChain->GetEntries();
    long int nbytes   = 0;

    std::cout << TAG << "Reading events...\n";
    for(long int jentry = 0; jentry < nentries; jentry++)
    {
        nbytes += fChain->GetEntry(jentry);
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

        if(read_extra_var)
        {
            ev.SetmuMomRec(muMomRec);
            ev.SetmuMomTrue(muMomTrue);
            ev.SetmuCosThetaRec(muCosThetaRec);
            ev.SetmuCosThetaTrue(muCosThetaTrue);
            ev.SetpMomRec(pMomRec);
            ev.SetpMomTrue(pMomTrue);
            ev.SetpCosThetaRec(pCosThetaRec);
            ev.SetpCosThetaTrue(pCosThetaTrue);
        }

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
            if(sample->GetSampleID() == cutBranch)
                sample->AddEvent(ev);
        }

        if(jentry % 2000 == 0 || jentry == (nentries - 1))
            pbar.Print(jentry, nentries - 1);
    }

    for(auto& sample : ana_samples)
        sample->PrintStats();
}
