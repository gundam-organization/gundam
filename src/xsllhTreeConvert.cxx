#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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

struct HL2TreeVar
{
    std::string reaction;
    std::string topology;
    std::string target;
    std::string nutype;
    std::string enu_true;
    std::string enu_reco;
    std::string weight;
    std::string D1True;
    std::string D1Reco;
    std::string D2True;
    std::string D2Reco;
};

struct HL2FileOpt
{
    std::string fname_input;
    std::string sel_tree;
    std::string tru_tree;
    unsigned int file_id;
    unsigned int num_branches;
    double pot_norm;
    std::vector<int> cuts;
    std::map<int, std::vector<int>> samples;

    HL2TreeVar sel_var;
    HL2TreeVar tru_var;
};

template <typename T>
HL2TreeVar ParseHL2Var(T json_obj, bool flag);

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

    std::cout << TAG << "Reading configuration options..." << std::endl;
    std::string out_fname = j["output"]["fname"];
    std::string out_seltree_name = j["output"]["sel_tree"];
    std::string out_trutree_name = j["output"]["tru_tree"];

    std::cout << TAG << "Out File: " << out_fname << std::endl
              << TAG << "Out Selection Tree: " << out_seltree_name << std::endl
              << TAG << "Out Truth Tree    : " << out_trutree_name << std::endl;

    TFile* out_file = TFile::Open(out_fname.c_str(), "RECREATE");
    TTree* out_seltree = new TTree(out_seltree_name.c_str(), out_seltree_name.c_str());
    TTree* out_trutree = new TTree(out_trutree_name.c_str(), out_trutree_name.c_str());

    const float mu_mass = 105.658374;
    int nutype, nutype_true;
    int reaction, reaction_true;
    int topology, topology_true;
    int target, target_true;
    int cut_branch;
    float enu_true, enu_reco;
    float q2_true, q2_reco;
    float D1True, D1Reco;
    float D2True, D2Reco;
    float weight, weight_true;

    out_seltree -> Branch("nutype", &nutype, "nutype/I");
    out_seltree -> Branch("reaction", &reaction, "reaction/I");
    out_seltree -> Branch("topology", &topology, "topology/I");
    out_seltree -> Branch("target", &target, "target/I");
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

    out_trutree -> Branch("nutype", &nutype_true, "nutype/I");
    out_trutree -> Branch("reaction", &reaction_true, "reaction/I");
    out_trutree -> Branch("topology", &topology_true, "topology/I");
    out_trutree -> Branch("target", &target_true, "target/I");
    out_trutree -> Branch("cut_branch", &cut_branch, "cut_branch/I");
    out_trutree -> Branch("enu_true", &enu_true, "enu_true/F");
    out_trutree -> Branch("q2_true", &q2_true, "q2_true/F");
    out_trutree -> Branch("D1True", &D1True, "D1True/F");
    out_trutree -> Branch("D2True", &D2True, "D2True/F");
    out_trutree -> Branch("weight", &weight_true, "weight/F");

    std::vector<HL2FileOpt> v_files;
    for(const auto& file : j["highland_files"])
    {
        if(file["use"])
        {
            HL2FileOpt f;
            f.fname_input = file["fname"];
            f.sel_tree = file["sel_tree"];
            f.tru_tree = file["tru_tree"];
            f.file_id = file["file_id"];
            f.num_branches = file["num_branches"];
            f.cuts = file["cut_level"].get<std::vector<int>>();
            f.pot_norm = file["pot_norm"];

            std::map<std::string, std::vector<int>> temp_json = file["samples"];
            for(const auto& kv : temp_json)
                f.samples.emplace(std::make_pair(std::stoi(kv.first), kv.second));

            f.sel_var = ParseHL2Var(file["sel_var"], true);
            f.tru_var = ParseHL2Var(file["tru_var"], false);

            v_files.emplace_back(f);
        }
    }

    for(const auto& file : v_files)
    {
        std::cout << TAG << "Reading file: " << file.fname_input << std::endl
                  << TAG << "File ID: " << file.file_id << std::endl
                  << TAG << "Selected tree: " << file.sel_tree << std::endl
                  << TAG << "Truth tree: " << file.tru_tree << std::endl
                  << TAG << "POT Norm: " << file.pot_norm << std::endl
                  << TAG << "Num. Branches: " << file.num_branches << std::endl;

        std::cout << TAG << "Branch to Sample mapping:" << std::endl;
        for(const auto& kv : file.samples)
        {
            std::cout << TAG << "Sample " << kv.first << ": ";
            for(const auto& b : kv.second)
                std::cout << b << " ";
            std::cout << std::endl;
        }

        TFile* hl2_file = TFile::Open(file.fname_input.c_str(), "READ");
        TTree* hl2_seltree = (TTree*)hl2_file -> Get(file.sel_tree.c_str());
        TTree* hl2_trutree = (TTree*)hl2_file -> Get(file.tru_tree.c_str());

        int accum_level[1][file.num_branches];

        hl2_seltree -> SetBranchAddress("accum_level", &accum_level);
        hl2_seltree -> SetBranchAddress(file.sel_var.nutype.c_str(), &nutype);
        hl2_seltree -> SetBranchAddress(file.sel_var.reaction.c_str(), &reaction);
        hl2_seltree -> SetBranchAddress(file.sel_var.topology.c_str(), &topology);
        hl2_seltree -> SetBranchAddress(file.sel_var.target.c_str(), &target);
        hl2_seltree -> SetBranchAddress(file.sel_var.D1Reco.c_str(), &D1Reco);
        hl2_seltree -> SetBranchAddress(file.sel_var.D2Reco.c_str(), &D2Reco);
        hl2_seltree -> SetBranchAddress(file.sel_var.D1True.c_str(), &D1True);
        hl2_seltree -> SetBranchAddress(file.sel_var.D2True.c_str(), &D2True);
        hl2_seltree -> SetBranchAddress(file.sel_var.enu_true.c_str(), &enu_true);
        hl2_seltree -> SetBranchAddress(file.sel_var.enu_reco.c_str(), &enu_reco);
        hl2_seltree -> SetBranchAddress(file.sel_var.weight.c_str(), &weight);

        long int npassed = 0;
        long int nevents = hl2_seltree -> GetEntries();
        std::cout << TAG << "Reading selected events tree." << std::endl
                  << TAG << "Num. events: " << nevents << std::endl;

        for(int i = 0; i < nevents; ++i)
        {
            hl2_seltree -> GetEntry(i);

            bool event_passed = false;
            for(const auto& kv : file.samples)
            {
                for(const auto& branch : kv.second)
                {
                    if(accum_level[0][branch] > file.cuts[branch])
                    {
                        cut_branch = kv.first;
                        event_passed = true;
                        npassed++;
                        break;
                    }
                }
            }

            float selmu_mom = D1Reco;
            float selmu_cos = D2Reco;
            float selmu_mom_true = D1True;
            float selmu_cos_true = D2True;

            double emu_true = std::sqrt(selmu_mom_true * selmu_mom_true + mu_mass * mu_mass);
            q2_true = 2.0 * enu_true * (emu_true - selmu_mom_true * selmu_cos_true)
                - mu_mass * mu_mass;

            double emu_reco = std::sqrt(selmu_mom * selmu_mom + mu_mass * mu_mass);
            q2_reco = 2.0 * enu_reco * (emu_reco - selmu_mom * selmu_cos)
                - mu_mass * mu_mass;

            weight *= file.pot_norm;

            if(event_passed)
                out_seltree -> Fill();

            if(i % 2000 == 0 || i == (nevents-1))
                pbar.Print(i, nevents-1);
        }
        std::cout << TAG << "Selected events passing cuts: " << npassed << std::endl;

        hl2_trutree -> SetBranchAddress(file.tru_var.nutype.c_str(), &nutype_true);
        hl2_trutree -> SetBranchAddress(file.tru_var.reaction.c_str(), &reaction_true);
        hl2_trutree -> SetBranchAddress(file.tru_var.topology.c_str(), &topology_true);
        hl2_trutree -> SetBranchAddress(file.tru_var.target.c_str(), &target_true);
        hl2_trutree -> SetBranchAddress(file.tru_var.D1True.c_str(), &D1True);
        hl2_trutree -> SetBranchAddress(file.tru_var.D2True.c_str(), &D2True);
        hl2_trutree -> SetBranchAddress(file.tru_var.enu_true.c_str(), &enu_true);
        hl2_trutree -> SetBranchAddress(file.tru_var.weight.c_str(), &weight_true);

        nevents = hl2_trutree -> GetEntries();
        std::cout << TAG << "Reading truth events tree." << std::endl
                  << TAG << "Num. events: " << nevents << std::endl;
        for(int i = 0; i < nevents; ++i)
        {
            hl2_trutree -> GetEntry(i);
            cut_branch = -1;

            float selmu_mom_true = D1True;
            float selmu_cos_true = D2True;

            double emu_true = std::sqrt(selmu_mom_true * selmu_mom_true + mu_mass * mu_mass);
            q2_true = 2.0 * enu_true * (emu_true - selmu_mom_true * selmu_cos_true)
                - mu_mass * mu_mass;

            weight *= file.pot_norm;
            out_trutree -> Fill();

            if(i % 2000 == 0 || i == (nevents-1))
                pbar.Print(i, nevents-1);
        }

        hl2_file -> Close();
    }

    out_file -> cd();
    out_file -> Write();
    out_file -> Close();
    std::cout << TAG << "Finished." << std::endl;

    return 0;
}

template <typename T>
HL2TreeVar ParseHL2Var(T j, bool reco_info)
{
    HL2TreeVar v;
    v.reaction = j["reaction"];
    v.topology = j["topology"];
    v.target   = j["target"];
    v.nutype   = j["nutype"];
    v.enu_true = j["enu_true"];
    v.weight = j["weight"];

    v.D1True = j["D1True"];
    v.D2True = j["D2True"];

    if(reco_info)
    {
        v.enu_reco = j["enu_reco"];
        v.D1Reco = j["D1Reco"];
        v.D2Reco = j["D2Reco"];
    }

    return v;
}
