//This is the code that actually reads int he MC tree and fills the event info.
//The tree should be produced by feeding a HL2 microtree into the treeconvert macro.

#include "AnyTreeMC.hh"

AnyTreeMC::AnyTreeMC(const std::string& file_name)
{
    std::string tree_name("selectedEvents");
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

long int AnyTreeMC::GetEntry(long int entry)
{
    // Read contents of entry.
    if(fChain == nullptr)
        return 0;
    else
        return fChain -> GetEntry(entry);
}

void AnyTreeMC::SetBranches()
{
    // Set branch addresses and branch pointers
    fChain -> SetBranchAddress("mectopology", &evtTopology, &b_evtTopology);
    fChain -> SetBranchAddress("reaction", &evtReaction, &b_evtReaction);
    fChain -> SetBranchAddress("D1True", &trueD1, &b_trueD1);
    fChain -> SetBranchAddress("D2True", &trueD2, &b_trueD2);
    fChain -> SetBranchAddress("cutBranch", &qesampleFinal, &b_qesampleFinal);
    fChain -> SetBranchAddress("D1Rec", &MainD1Glb, &b_MainD1Glb);
    fChain -> SetBranchAddress("D2Rec", &MainD2, &b_MainD2);
    fChain -> SetBranchAddress("Enureco", &MainRecEneGlb, &b_MainRecEneGlb);
    fChain -> SetBranchAddress("Enutrue", &TrueEnergy, &b_TrueEnergy);
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

void AnyTreeMC::GetEvents(std::vector<AnaSample*> ana_samples)
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
        ev.SetSampleType(qesampleFinal);
        int evtTopo = evtTopology; //For my analysis 0 CC0pi0p, 1 CC0pi1p, 2 CC0pinp, 3 CC1pi, 4 CCOther, 5 backg(NC+antinu), 7 OOFV
        //cout << "Evt Topology is " << evtTopo << endl;
        ev.SetTopology(evtTopology); // mectopology (i.e. CC0Pi,CC1Pi etc)
        ev.SetReaction(evtReaction); // reaction (i.e. CCQE,CCRES etc)
        ev.SetTrueEnu(TrueEnergy/1000.0);   //MeV - ->  GeV
        ev.SetRecEnu(MainRecEneGlb/1000.0); //MeV - ->  GeV
        ev.SetTrueD1trk(trueD1);
        ev.SetRecD1trk(MainD1Glb);
        ev.SetTrueD2trk(trueD2);
        ev.SetRecD2trk(MainD2);
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

        for(auto& sample : ana_samples)
        {
            if(sample -> GetSampleType() == qesampleFinal)
                sample -> AddEvent(ev);
        }
    }

    for(auto& sample : ana_samples)
        sample -> PrintStats();
}
