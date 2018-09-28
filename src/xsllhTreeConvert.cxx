#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include <TFile.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TTree.h>

#include "json.hpp"
using json = nlohmann::json;

#include "ColorOutput.hh"
#include "ProgressBar.hh"

int main(int argc, char** argv)
{
    const std::string TAG = color::GREEN_STR + "[xsTreeConvert]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR
                            + "[ERROR]: " + color::RESET_STR;

    ProgressBar pbar(60, "#");
    pbar.SetRainbow();
    pbar.SetPrefix(std::string(TAG + "Reading Events "));

    std::cout << "-------------------------------------------------\n"
              << TAG << "Welcome to the Super-xsLLh Tree Converter.\n"
              << TAG << "Initializing the tree machinery..." << std::endl;

    std::string json_file;

    char option;
    while((option = getopt(argc, argv, "j:h")) != -1)
    {
        switch(option)
        {
            case 'j':
                json_file = optarg;
                break;
            case 'h':
                std::cout << "USAGE: "
                          << argv[0] << "\nOPTIONS:\n"
                          << "-j : JSON input\n";
            default:
                return 0;
        }
    }

    std::fstream f;
    f.open(json_file, std::ios::in);
    std::cout << TAG << "Opening " << json_file << std::endl;
    if(!f.is_open())
    {
        std::cout << ERR << "Unable to open JSON configure file." << std::endl;
        return 1;
    }

    json j;
    f >> j;

    std::string nd2_fname = j["INGRID"]["fname"];
    std::string nd2_seltree_name = j["INGRID"]["seltree"];
    std::string nd2_trutree_name = j["INGRID"]["truetree"];
    std::string nd5_fname = j["ND280"]["fname"];
    std::string nd5_seltree_name = j["ND280"]["seltree"];
    std::string nd5_trutree_name = j["ND280"]["truetree"];
    std::string out_fname = j["output"]["fname"];
    std::string out_seltree_name = j["output"]["seltree"];
    std::string out_trutree_name = j["output"]["truetree"];

    const int nd5_samples = j["ND280"]["nsamples"];
    std::vector<int> cut_level = j["ND280"]["cut_level"].get<std::vector<int>>();

    std::cout << TAG << "ND2 File: " << nd2_fname << std::endl
              << TAG << "ND2 Selection Tree: " << nd2_seltree_name << std::endl
              << TAG << "ND2 Truth Tree    : " << nd2_trutree_name << std::endl;
    std::cout << TAG << "ND5 File: " << nd5_fname << std::endl
              << TAG << "ND5 Selection Tree: " << nd5_seltree_name << std::endl
              << TAG << "ND5 Truth Tree    : " << nd5_trutree_name << std::endl;
    std::cout << TAG << "Out File: " << out_fname << std::endl
              << TAG << "Out Selection Tree: " << out_seltree_name << std::endl
              << TAG << "Out Truth Tree    : " << out_trutree_name << std::endl;

    TFile* nd2_file = TFile::Open(nd2_fname.c_str(), "READ");
    TTree* nd2_seltree = (TTree*)nd2_file -> Get(nd2_seltree_name.c_str());
    TTree* nd2_trutree = (TTree*)nd2_file -> Get(nd2_trutree_name.c_str());

    TFile* nd5_file = TFile::Open(nd5_fname.c_str(), "READ");
    TTree* nd5_seltree = (TTree*)nd5_file -> Get(nd5_seltree_name.c_str());
    TTree* nd5_trutree = (TTree*)nd5_file -> Get(nd5_trutree_name.c_str());

    TFile* out_file = TFile::Open(out_fname.c_str(), "RECREATE");
    TTree* out_seltree = new TTree(out_seltree_name.c_str(), out_seltree_name.c_str());
    TTree* out_trutree = new TTree(out_trutree_name.c_str(), out_trutree_name.c_str());

    int accum_level[1][nd5_samples];
    int nutype, nutype_true;
    int reaction, reaction_true;
    int topology, topology_true;
    int cut_branch;
    float enu_true, enu_reco;
    float q2_true, q2_reco;
    float D1True, D1Reco;
    float D2True, D2Reco;
    float weight, weight_true;

    const float mu_mass = 105.658374;

    float selmu_mom, selmu_cos;
    float selmu_mom_range;
    float selp_mom, selp_cos;
    float selp_mom_range;
    float selmu_mom_true, selmu_cos_true;
    float selp_mom_true, selp_cos_true;

    nd5_seltree -> SetBranchAddress("accum_level", &accum_level);
    nd5_seltree -> SetBranchAddress("nutype", &nutype);
    nd5_seltree -> SetBranchAddress("reaction", &reaction);
    nd5_seltree -> SetBranchAddress("mectopology", &topology);
    nd5_seltree -> SetBranchAddress("selp_mom", &selp_mom);
    nd5_seltree -> SetBranchAddress("selp_costheta", &selp_cos);
    nd5_seltree -> SetBranchAddress("selp_mom_range_oarecon", &selp_mom_range);
    nd5_seltree -> SetBranchAddress("truep_truemom", &selp_mom_true);
    nd5_seltree -> SetBranchAddress("truep_truecostheta", &selp_cos_true);
    nd5_seltree -> SetBranchAddress("selmu_mom", &selmu_mom);
    nd5_seltree -> SetBranchAddress("selmu_costheta", &selmu_cos);
    nd5_seltree -> SetBranchAddress("selmu_mom_range_oarecon", &selmu_mom_range);
    nd5_seltree -> SetBranchAddress("truelepton_mom", &selmu_mom_true);
    nd5_seltree -> SetBranchAddress("truelepton_costheta", &selmu_cos_true);
    nd5_seltree -> SetBranchAddress("nu_trueE", &enu_true);
    nd5_seltree -> SetBranchAddress("weight_syst_total", &weight);

    out_seltree -> Branch("nutype", &nutype, "nutype/I");
    out_seltree -> Branch("reaction", &reaction, "reaction/I");
    out_seltree -> Branch("topology", &topology, "topology/I");
    out_seltree -> Branch("cut_branch", &cut_branch, "cut_branch/I");
    out_seltree -> Branch("enu_true", &enu_true, "enu_true/F");
    out_seltree -> Branch("enu_reco", &enu_reco, "enu_reco/F");
    out_seltree -> Branch("q2_true", &q2_true, "q2_true/F");
    out_seltree -> Branch("q2_reco", &q2_reco, "q2_reco/F");
    out_seltree -> Branch("D1True", &D1True, "D1True/F");
    out_seltree -> Branch("D1Reco", &D1Reco, "D1Reco/F");
    out_seltree -> Branch("D2True", &D2True, "D2True/F");
    out_seltree -> Branch("D2Reco", &D2Reco, "D2Reco/F");
    out_seltree -> Branch("weight", &weight, "weight/F");

    nd5_trutree -> SetBranchAddress("nutype", &nutype_true);
    nd5_trutree -> SetBranchAddress("reaction", &reaction_true);
    nd5_trutree -> SetBranchAddress("mectopology", &topology_true);
    nd5_trutree -> SetBranchAddress("truehm_proton_truemom", &selp_mom_true);
    nd5_trutree -> SetBranchAddress("truehm_proton_truecth", &selp_cos_true);
    nd5_trutree -> SetBranchAddress("truelepton_mom", &selmu_mom_true);
    nd5_trutree -> SetBranchAddress("truelepton_costheta", &selmu_cos_true);
    nd5_trutree -> SetBranchAddress("nu_trueE", &enu_true);
    nd5_trutree -> SetBranchAddress("weight", &weight_true);

    out_trutree -> Branch("nutype", &nutype_true, "nutype/I");
    out_trutree -> Branch("reaction", &reaction_true, "reaction/I");
    out_trutree -> Branch("topology", &topology_true, "topology/I");
    out_trutree -> Branch("cut_branch", &cut_branch, "cut_branch/I");
    out_trutree -> Branch("enu_true", &enu_true, "enu_true/F");
    out_trutree -> Branch("q2_true", &q2_true, "q2_true/F");
    out_trutree -> Branch("D1True", &D1True, "D1True/F");
    out_trutree -> Branch("D2True", &D2True, "D2True/F");
    out_trutree -> Branch("weight", &weight_true, "weight/F");

    long int nevents = nd5_seltree -> GetEntries();
    for(int i = 0; i < nevents; ++i)
    {
        nd5_seltree -> GetEntry(i);

        bool event_passed = false;
        int branches_passed = 0;
        for(int s = 0; s < nd5_samples; ++s)
        {
            if(accum_level[0][s] > cut_level[s])
            {
                cut_branch = s;
                event_passed = true;
                branches_passed++;
            }
        }
        enu_reco = enu_true;
        double emu_true = std::sqrt(selmu_mom_true * selmu_mom_true + mu_mass * mu_mass);
        q2_true = 2.0 * enu_true * (emu_true - selmu_mom_true * selmu_cos_true)
                         - mu_mass * mu_mass;

        double emu_reco = std::sqrt(selmu_mom * selmu_mom + mu_mass * mu_mass);
        q2_reco = 2.0 * enu_reco * (emu_reco - selmu_mom * selmu_cos)
                         - mu_mass * mu_mass;
        /*
        if(cut_branch == 3)
        {
            D1True = selmu_mom_true;
            D1Reco = selmu_mom_range;
            D2True = selmu_cos_true;
            D2Reco = selmu_cos;
        }
        else
        {
            D1True = selmu_mom_true;
            D1Reco = selmu_mom;
            D2True = selmu_cos_true;
            D2Reco = selmu_cos;
        }
        */
            D1True = selmu_mom_true;
            D1Reco = selmu_mom;
            D2True = selmu_cos_true;
            D2Reco = selmu_cos;

        if(event_passed && branches_passed == 1)
            out_seltree -> Fill();

        if(i % 2000 == 0 || i == (nevents-1))
            pbar.Print(i, nevents-1);
    }

    nevents = nd5_trutree -> GetEntries();
    for(int i = 0; i < nevents; ++i)
    {
        nd5_trutree -> GetEntry(i);
        cut_branch = -1;

        double emu_true = std::sqrt(selmu_mom_true * selmu_mom_true + mu_mass * mu_mass);
        q2_true = 2.0 * enu_true * (emu_true - selmu_mom_true * selmu_cos_true)
                         - mu_mass * mu_mass;

        D1True = selmu_mom_true;
        D2True = selmu_cos_true;

        out_trutree -> Fill();

        if(i % 2000 == 0 || i == (nevents-1))
            pbar.Print(i, nevents-1);
    }

    float muang, pang;
    float nuE, range;
    float opening;
    int file_index, inttype;
    int mupdg, ppdg;
    int npioncount;
    int nprotoncount;

    nd2_seltree->SetBranchAddress("muang", &muang);
    nd2_seltree->SetBranchAddress("pang", &pang);
    nd2_seltree->SetBranchAddress("opening", &opening);
    nd2_seltree->SetBranchAddress("nuE", &nuE);
    nd2_seltree->SetBranchAddress("fileIndex", &file_index);
    nd2_seltree->SetBranchAddress("inttype", &inttype);
    nd2_seltree->SetBranchAddress("mupdg", &mupdg);
    nd2_seltree->SetBranchAddress("ppdg", &ppdg);
    nd2_seltree->SetBranchAddress("range", &range);
    nd2_seltree->SetBranchAddress("npioncount", &npioncount);
    nd2_seltree->SetBranchAddress("nprotoncount", &nprotoncount);

    TRandom3 rng;

    //nevents = nd2_seltree -> GetEntries();
    int sevents = 2000000;
    nevents = 2030000;

    for(int i = sevents; i < nevents; ++i)
    {
        nd2_seltree -> GetEntry(i);
        if(i % 2000 == 0 || i == (nevents-1))
            pbar.Print(i, nevents-1);

        if(file_index != 1)
            continue;

        enu_reco = nuE * 1000;
        enu_true = nuE * 1000;
        nutype = 14;
        nutype_true = 14;
        cut_branch = 10;
        weight = 1.0;

        if(inttype == 1)
            reaction = 0;
        else if(inttype == 2)
            reaction = 9;
        else if(inttype == 11 || inttype == 12 || inttype == 13)
            reaction = 1;
        else if(inttype == 21 || inttype == 26)
            reaction = 2;
        else if(inttype == 16)
            reaction = 3;
        else if(inttype > 30 && inttype < 47)
            reaction = 4;
        else if(inttype < 0)
            reaction = 5;
        else
            reaction = -1;

        if(file_index == 1 && npioncount == 0 && mupdg == 13 && ppdg == 2212 && nprotoncount == 1)
        {
            topology = 1;
            weight = 1.0;
        }
        else if(file_index == 1 && npioncount == 0 && mupdg == 13 && ppdg == 2212 && nprotoncount > 1)
        {
            topology = 2;
            weight = 1.0;
        }
        else if(file_index == 1 && npioncount == 1 && mupdg == 13 && ppdg == 211 && nprotoncount == 0)
            topology = 3;
        else
            continue;

        double muang_true = TMath::DegToRad() * muang;
        double pang_true = TMath::DegToRad() * pang;
        double muang_reco = muang_true * rng.Gaus(1, 0.1);
        double pang_reco = pang_true * rng.Gaus(1, 0.1);

        selmu_cos = TMath::Cos(muang_reco);
        selmu_cos_true = TMath::Cos(muang_true);

        selp_cos = TMath::Cos(pang_reco);
        selp_cos_true = TMath::Cos(pang_true);

        selmu_mom_true = (range * 0.0114127 + 0.230608) * 1000.0;
        selmu_mom = ((range * rng.Gaus(1, 0.1)) * 0.0114127 + 0.230608) * 1000.0;

        double emu_true = std::sqrt(selmu_mom_true * selmu_mom_true + mu_mass * mu_mass);
        q2_true = 2.0 * enu_true * (emu_true - selmu_mom_true * selmu_cos_true)
                         - mu_mass * mu_mass;

        double emu_reco = std::sqrt(selmu_mom * selmu_mom + mu_mass * mu_mass);
        q2_reco = 2.0 * enu_reco * (emu_reco - selmu_mom * selmu_cos)
                         - mu_mass * mu_mass;

        D1True = selmu_mom_true;
        D1Reco = selmu_mom;
        D2True = selmu_cos_true;
        D2Reco = selmu_cos;

        out_seltree -> Fill();

        cut_branch = -2;
        reaction_true = reaction;
        topology_true = topology;
        nutype_true = nutype;
        weight_true = weight;

        out_trutree -> Fill();
    }

    out_file -> cd();
    out_file -> Write();

    nd2_file -> Close();
    nd5_file -> Close();
    out_file -> Close();

    return 0;
}
