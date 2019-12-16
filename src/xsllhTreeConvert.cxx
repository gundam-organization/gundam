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

// Structure that holds the variable names for the Highland2 file:
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
    std::map<std::string, std::vector<int>> D1Reco_multi;
    bool use_D1Reco_multi;
    std::string D2True;
    std::string D2Reco;
};

// Structure that holds the file information for the Highland2 file (including variable names):
struct HL2FileOpt
{
    std::string fname_input;
    std::string sel_tree;
    std::string tru_tree;
    unsigned int file_id;
    int beammode;
    int nutype_selection;
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
    // Define colors and strings for info and error messages:
    const std::string TAG = color::GREEN_STR + "[xsTreeConvert]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR + "[ERROR]: " + color::RESET_STR;

    // Progress bar for reading in events from the ROOT file:
    ProgressBar pbar(60, "#");
    pbar.SetRainbow();
    pbar.SetPrefix(std::string(TAG + "Reading Events "));

    // Print welcome message:
    std::cout << "-------------------------------------------------\n"
              << TAG << "Welcome to the Super-xsLLh Tree Converter.\n"
              << TAG << "Initializing the tree machinery..." << std::endl;

    // .json config file that will be parsed from the command line:
    std::string json_file;

    // Initialize json_file with the name parsed from the command line using -j. Print USAGE and exit when -h is used:
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

    // Read in .json config file:
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

    // Print names of the output file and the selection/truth tree names:
    std::cout << TAG << "Out File: " << out_fname << std::endl
              << TAG << "Out Selection Tree: " << out_seltree_name << std::endl
              << TAG << "Out Truth Tree    : " << out_trutree_name << std::endl;

    // Create the output file and define its ROOT trees. If it already exists, it will be overwritten:
    TFile* out_file = TFile::Open(out_fname.c_str(), "RECREATE");
    TTree* out_seltree = new TTree(out_seltree_name.c_str(), out_seltree_name.c_str());
    TTree* out_trutree = new TTree(out_trutree_name.c_str(), out_trutree_name.c_str());

    // Declare some variables that will hold the values written to the output ROOT file:
    const float mu_mass = 105.658374;
    int nutype, nutype_true;
    int reaction, reaction_true;
    int topology, topology_true;
    int target, target_true;
    int cut_branch;
    int beammode;
    int nutype_selection;
    float enu_true, enu_reco;
    float q2_true, q2_reco;
    float D1True, D1Reco;
    float D2True, D2Reco;
    float weight, weight_true;

    // Add branches to output ROOT file:
    out_seltree -> Branch("nutype", &nutype, "nutype/I");
    out_seltree -> Branch("reaction", &reaction, "reaction/I");
    out_seltree -> Branch("topology", &topology, "topology/I");
    out_seltree -> Branch("target", &target, "target/I");
    out_seltree -> Branch("cut_branch", &cut_branch, "cut_branch/I");
    out_seltree -> Branch("beammode", &beammode, "beammode/I");
    out_seltree -> Branch("nutype_selection", &nutype_selection, "nutype_selection/I");
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
    out_trutree -> Branch("beammode", &beammode, "beammode/I");
    out_trutree -> Branch("nutype_selection", &nutype_selection, "nutype_selection/I");
    out_trutree -> Branch("enu_true", &enu_true, "enu_true/F");
    out_trutree -> Branch("q2_true", &q2_true, "q2_true/F");
    out_trutree -> Branch("D1True", &D1True, "D1True/F");
    out_trutree -> Branch("D2True", &D2True, "D2True/F");
    out_trutree -> Branch("weight", &weight_true, "weight/F");

    // This vector will store the file information for all files specified in the .json config file (including variable names):
    std::vector<HL2FileOpt> v_files;

    // Loop over all files specified in the .json config file and add their inforamtion to v_files:
    for(const auto& file : j["highland_files"])
    {
        if(file["use"])
        {
            // If there is a "flist" key present in the .json config file, we add from the corresponding text file:
            if(file.find("flist") != file.end())
            {
                // Text file containing all files to be read in:
                std::ifstream in(file["flist"]);
                std::string filename;

                // Loop over lines of text file with the file list and add contents to the v_files vector:
                while(std::getline(in, filename))
                {
                    HL2FileOpt f;
                    f.fname_input = filename;
                    f.sel_tree = file["sel_tree"];
                    f.tru_tree = file["tru_tree"];
                    f.file_id = file["file_id"];
                    f.beammode = file["beammode"];
                    f.nutype_selection = file["nutype_selection"];
                    f.num_branches = file["num_branches"];
                    f.cuts = file["cut_level"].get<std::vector<int>>();
                    f.pot_norm = file["pot_norm"];

                    std::map<std::string, std::vector<int>> temp_json = file["samples"];
                    for(const auto& kv : temp_json)
                        f.samples.emplace(std::make_pair(std::stoi(kv.first), kv.second));

                    // Read out the json objects for "sel_var" and "tru_var":
                    f.sel_var = ParseHL2Var(file["sel_var"], true);
                    f.tru_var = ParseHL2Var(file["tru_var"], false);

                    v_files.emplace_back(f);
                }
            }

            // Otherwise we use the "fname" key to get the name of the file which is to be read in:
            else
            {
                HL2FileOpt f;
                f.fname_input = file["fname"];
                f.sel_tree = file["sel_tree"];
                f.tru_tree = file["tru_tree"];
                f.file_id = file["file_id"];
                f.beammode = file["beammode"];
                f.nutype_selection = file["nutype_selection"];
                f.num_branches = file["num_branches"];
                f.cuts = file["cut_level"].get<std::vector<int>>();
                f.pot_norm = file["pot_norm"];

                std::map<std::string, std::vector<int>> temp_json = file["samples"];
                for(const auto& kv : temp_json)
                    f.samples.emplace(std::make_pair(std::stoi(kv.first), kv.second));

                // Read out the json objects for "sel_var" and "tru_var":
                f.sel_var = ParseHL2Var(file["sel_var"], true);
                f.tru_var = ParseHL2Var(file["tru_var"], false);

                v_files.emplace_back(f);
            }
        }
    }

    // Loop over all the files that were read in:
    for(const auto& file : v_files)
    {
        // Some info messages about each file:
        std::cout << TAG << "Reading file: " << file.fname_input << std::endl
                  << TAG << "File ID: " << file.file_id << std::endl
                  << TAG << "Selected tree: " << file.sel_tree << std::endl
                  << TAG << "Truth tree: " << file.tru_tree << std::endl
                  << TAG << "POT Norm: " << file.pot_norm << std::endl
                  << TAG << "Beam mode: " << file.beammode << std::endl
                  << TAG << "Nu-type Selection: " << file.nutype_selection << std::endl
                  << TAG << "Num. Branches: " << file.num_branches << std::endl;

        std::cout << TAG << "Branch to Sample mapping:" << std::endl;
        for(const auto& kv : file.samples)
        {
            std::cout << TAG << "Sample " << kv.first << ": ";
            for(const auto& b : kv.second)
                std::cout << b << " ";
            std::cout << std::endl;
        }

        // Beam mode was previously read in from the .json config file:
        beammode = file.beammode;
        nutype_selection = file.nutype_selection;

        // Open input ROOT file to read it and get the selected and truth trees:
        TFile* hl2_file = TFile::Open(file.fname_input.c_str(), "READ");
        TTree* hl2_seltree = (TTree*)hl2_file -> Get(file.sel_tree.c_str());
        TTree* hl2_trutree = (TTree*)hl2_file -> Get(file.tru_tree.c_str());

        // Set the branch addresses for the selected tree to the previously declared variables:
        int accum_level[1][file.num_branches];

        hl2_seltree -> SetBranchAddress("accum_level", &accum_level);
        hl2_seltree -> SetBranchAddress(file.sel_var.nutype.c_str(), &nutype);
        hl2_seltree -> SetBranchAddress(file.sel_var.reaction.c_str(), &reaction);
        hl2_seltree -> SetBranchAddress(file.sel_var.topology.c_str(), &topology);
        hl2_seltree -> SetBranchAddress(file.sel_var.target.c_str(), &target);

        // If the use_D1Reco_multi flag has been set to true, different selection branches will have different D1Reco variables:
        std::vector<float> D1Reco_vector(file.sel_var.D1Reco_multi.size());
        if(file.sel_var.use_D1Reco_multi)
        {
            std::cout << TAG << "Using different D1Reco variables depending on the branch." << std::endl;
            int iter = 0;

            // Loop over all entries of the D1Reco json object:
            for(const auto& kv : file.sel_var.D1Reco_multi)
            {
                hl2_seltree -> SetBranchAddress(kv.first.c_str(), &D1Reco_vector[iter]);
                ++iter;
            }
        }

        // Otherwise the same D1Reco variable will be used for all selection branches:
        else
        {
            std::cout << TAG << "Using the same D1Reco variables for all branches." << std::endl;
            hl2_seltree -> SetBranchAddress(file.sel_var.D1Reco.c_str(), &D1Reco);
        }

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

        // Loop over all events in the input ROOT file in the selected tree:
        for(int i = 0; i < nevents; ++i)
        {
            hl2_seltree -> GetEntry(i);

            bool event_passed = false;

            // Loop over all samples specified in .json config file:
            for(const auto& kv : file.samples)
            {
                // Loop over all branches in current sample:
                for(const auto& branch : kv.second)
                {
                    // Event passed if its accum_level is higher than the given cut for this branch:
                    if(accum_level[0][branch] > file.cuts[branch])
                    {
                        cut_branch = kv.first;
                        event_passed = true;
                        npassed++;
                        break;
                    }
                }
            }

            // If we are using multiple variables for D1Reco depending on the branch, we set the D1Reco variable to the value of D1Reco of the current branch:
            if(file.sel_var.use_D1Reco_multi)
            {
                int it = 0;
                for(const auto& kv : file.sel_var.D1Reco_multi)
                {
                    if(std::find(kv.second.begin(), kv.second.end(), cut_branch) != kv.second.end())
                    {
                        D1Reco = D1Reco_vector[it];
                        break;
                    }
                    ++it;
                }
            }

            // Calculate muon energies and q2:
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

            // If the event passed the cuts, we fill the output ROOT file with the variables we defined:
            if(event_passed)
                out_seltree -> Fill();

            // Update progress bar:
            if(i % 2000 == 0 || i == (nevents-1))
                pbar.Print(i, nevents-1);
        }
        std::cout << TAG << "Selected events passing cuts: " << npassed << std::endl;

        // Set the branch addresses for the true tree to the previously declared variables:
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

        // Loop over all events in the input ROOT file in the truth tree:
        for(int i = 0; i < nevents; ++i)
        {
            hl2_trutree -> GetEntry(i);
            cut_branch = -1;

            float selmu_mom_true = D1True;
            float selmu_cos_true = D2True;

            // Calculate muon energy:
            double emu_true = std::sqrt(selmu_mom_true * selmu_mom_true + mu_mass * mu_mass);
            q2_true = 2.0 * enu_true * (emu_true - selmu_mom_true * selmu_cos_true)
                - mu_mass * mu_mass;

            weight_true *= file.pot_norm;
            out_trutree -> Fill();

            // Update progress bar:
            if(i % 2000 == 0 || i == (nevents-1))
                pbar.Print(i, nevents-1);
        }
        hl2_file -> Close();
    }

    // Write to output file and close:
    out_file -> cd();
    out_file -> Write();
    out_file -> Close();
    std::cout << TAG << "Finished." << std::endl;

    return 0;
}

// Reads out json object from .json config file:
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

    // If flag to include reconstructed data is set to true:
    if(reco_info)
    {
        v.enu_reco = j["enu_reco"];

        // If the "D1Reco" entry is a string, the same variable will be used for D1Reco for all branches of the selection:
        if(j["D1Reco"].is_string())
        {
            v.D1Reco = j["D1Reco"];
            v.use_D1Reco_multi = false;
        }

        // If the "D1Reco" entry is a json object, different variables can be used for D1Reco depending on the branch of the selection:
        else if(j["D1Reco"].is_object())
            for(auto& el : j["D1Reco"].items())
            {
                std::vector<int> tmp_vector = el.value();
                v.D1Reco_multi.emplace(el.key(), tmp_vector);
                v.use_D1Reco_multi = true;
            }

        v.D2Reco = j["D2Reco"];
    }
    return v;
}
