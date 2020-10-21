// This is the code that actually reads in the MC tree and fills the event info.
// The tree should be produced by feeding a HL2 microtree into the treeconvert macro.

#include "AnaTreeMC.hh"
#include "TTreeFormula.h"
#include "GenericToolbox.h"
#include "Logger.h"

AnaTreeMC::AnaTreeMC(const std::string& file_name, const std::string& tree_name, bool extra_var)
    : read_extra_var(extra_var)
{
    Logger::setUserHeaderStr("[AnaTreeMC]");
    fChain = new TChain(tree_name.c_str());
    fChain->Add(file_name.c_str());
    SetBranches();
}

AnaTreeMC::~AnaTreeMC()
{
    if(fChain != nullptr)
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
    fChain->SetBranchAddress("beammode", &beammode);
    fChain->SetBranchAddress("cut_branch", &sample);
    fChain->SetBranchAddress("topology", &topology);
    fChain->SetBranchAddress("reaction", &reaction);
    fChain->SetBranchAddress("target", &target);
    fChain->SetBranchAddress("D1True", &D1True);
    fChain->SetBranchAddress("D1Reco", &D1Reco);
    fChain->SetBranchAddress("D2True", &D2True);
    fChain->SetBranchAddress("D2Reco", &D2Reco);
    fChain->SetBranchAddress("q2_true", &q2_true);
    fChain->SetBranchAddress("q2_reco", &q2_reco);
    fChain->SetBranchAddress("enu_true", &enu_true);
    fChain->SetBranchAddress("enu_reco", &enu_reco);
    fChain->SetBranchAddress("weight", &weight);

    if(read_extra_var)
    {
        //Put extra variables here.
    }
}

void AnaTreeMC::GetEvents(std::vector<AnaSample*>& ana_samples,
                          const std::vector<SignalDef>& v_signal, const bool evt_type)
{
    if(fChain == nullptr || ana_samples.empty())
        return;

    std::vector<TTreeFormula*> additionalCutsList;
    for(size_t iSample = 0 ; iSample < ana_samples.size() ; iSample++){
        additionalCutsList.emplace_back(
            new TTreeFormula(
                Form("additional_cuts_%i", int(iSample)),
                ana_samples[iSample]->GetAdditionalCuts().c_str(),
                fChain
                )
            );
        additionalCutsList.back()->SetTree(fChain);
    }

    ProgressBar pbar(60, "#");
    pbar.SetRainbow();
    pbar.SetPrefix(std::string(TAG + "Reading Events "));

    int nentries = fChain->GetEntries();
    int nbytes   = 0;

    LogInfo << "Reading events (" << nentries << ")..." << std::endl;
    std::string progressBarPrefix = LogInfo.getPrefixString() + "Reading events...";

    // Loop over all events:
    for( int jEntry = 0; jEntry < nentries; jEntry++ )
    {
        GenericToolbox::displayProgressBar(jEntry, nentries, progressBarPrefix);
        nbytes += fChain->GetEntry(jEntry);
        AnaEvent ev(jEntry);
        ev.DumpTreeEntryContent(fChain);
        ev.SetTrueEvent(evt_type);
        ev.SetFlavor(nutype);
        ev.SetBeamMode(beammode);
        ev.SetSampleType(sample);
        ev.SetTopology(topology); // mectopology (i.e. CC0Pi,CC1Pi etc)
        ev.SetReaction(reaction); // reaction (i.e. CCQE,CCRES etc)
        ev.SetTarget(target);
        ev.SetTrueEnu(enu_true);
        ev.SetRecoEnu(enu_reco);
        ev.SetTrueD1(D1True);
        ev.SetRecoD1(D1Reco);
        ev.SetTrueD2(D2True);
        ev.SetRecoD2(D2Reco);
        ev.SetEvWght(weight);
        ev.SetEvWghtMC(weight);
        ev.SetQ2True(q2_true);
        ev.SetQ2Reco(q2_reco);

        if(read_extra_var)
        {
            //Put extra variables here.
        }

        int signal_type = 0;

        // Loop over all sets of temlate parameters as defined in the .json config file in the "template_par" entry for this detector:
        for(const auto& sd : v_signal)
        {
            bool sig_passed = true;

            // Loop over all the different signal definitions for this set of template parameters (e.g. topology, target, nutype, etc.):
            for(const auto& kv : sd.definition)
            {
                bool var_passed = false;
                
                // Loop over all the values for the current signal definition (e.g. all the different topology integers):
                for(const auto& val : kv.second)
                {
                    if(ev.GetEventVar(kv.first) == val)
                        var_passed = true;
                }
                sig_passed = sig_passed && var_passed;
            }
            if(sig_passed)
            {
                ev.SetSignalType(signal_type);
                ev.SetSignalEvent();
                break;
            }
            signal_type++;
        }

        for(size_t iSample = 0 ; iSample < ana_samples.size() ; iSample++){
            if(ana_samples[iSample]->GetSampleID() == sample){

                fChain->SetNotify(additionalCutsList[iSample]);
                for(int jInstance = 0; jInstance < additionalCutsList[iSample]->GetNdata(); jInstance++) {
                    if (additionalCutsList[iSample]->EvalInstance(jInstance) ) {
                        ana_samples[iSample]->AddEvent(ev);
                        break;
                    }
                }

            }
        }

//        if(jEntry % 2000 == 0 || jEntry == (nentries - 1))
//            pbar.Print(jEntry, nentries - 1);
    }

    for(auto& sample : ana_samples)
        sample->PrintStats();

    for(auto& additionalCuts : additionalCutsList){
        delete additionalCuts;
    }
}
