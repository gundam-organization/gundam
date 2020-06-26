#include <string>
#include <iostream>
#include <sstream>
#include <TFile.h>
#include <TTree.h>
#include <TLeaf.h>
#include <TGraph.h>
#include <TClonesArray.h>
#include <TSpline.h>
#include <TError.h>
#include <TVectorT.h>
#include <TH2D.h>

#include <FitStructs.hh>
#include <fstream>
#include <GenericToolbox.h>

const std::string ERROR    = "\033[1;31m[xsllhGenWeightsBinner] \033[00m";
const std::string INFO  = "\033[1;32m[xsllhGenWeightsBinner] \033[00m";
const std::string WARNING   = "\033[1;33m[xsllhGenWeightsBinner] \033[00m";
const std::string ALERT = "\033[1;35m[xsllhGenWeightsBinner] \033[00m";


//! Global Variables
std::string __json_config_path__;
std::string __genweights_file_path__;
std::string __genweights_file_list_path__;
std::string __binning_file_path__;
std::string __tree_converter_file_path__;
std::string __covariance_matrix_file_path__;

std::vector<std::string> __genweights_file_path_list__;
std::vector<std::string> __systematic_Xsec_spline_names_list__;
std::map<std::string, std::string> __from_Xsec_to_genWeights__;
std::vector<std::string> __relative_variation_component_list__;
std::map<std::string, int> __samples_name_to_index__;
std::map<std::string, int> __reaction_name_to_index__;
std::map<std::string, double> __parameter_nominal_values_map__;
std::map<std::string, double> __parameter_error_map__;

TFile* __input_genWeights_tfile__;
TTree* __sample_sum__;
TTree* __flattree__;

TFile* __tree_converter_tfile__;
TTree* __tree_converter_reco_ttree__;

bool __tree_sync_cache_has_been_build__;
std::map<int, int> __tree_sync_cache__;

bool __tree_converter_entries_have_been_mapped__ = false;
std::map<int, Int_t> __tree_converter_entries_mapping__;

std::map<std::string, TFile*> __output_tfile_splines__;

std::vector<xsllh::FitBin> __bin_edges__;

std::vector<std::string> __missing_splines__;
std::vector<std::string> __systematics_norm_splines_names_list__;
std::vector<std::string> __systematics_names_list__;

std::vector<int> __samples_list__;
std::vector<int> __reactions_list__;

int __current_event_sample__;
int __current_event_reaction__;
int __current_event_bin__;
int __last_event_id_found__ = 0;

std::string __command_line__;
int __argc__;
char **__argv__;


//! Local Functions
std::string remind_usage();
void reset_parameters();
void get_user_parameters();
void initialize_objects();
void destroy_objects();

void get_list_of_parameters();
void create_norm_splines();
void create_norm_splines();
void generate_json_config_file();
void regenarte_covariance_matrix_file();
void read_genweights_files_list();

void define_binning(std::string binning_file_);
int identify_event_bin(int entry_);

void build_tree_sync_cache();
void map_tree_converter_entries();
void fill_component_mapping();

void fill_parameter_nominal_values();
double from_relative_par_value_to_absolute(std::string par_name_,
                                           double deviation_param_value_);

bool do_syst_param_has_relative_X_scale(std::string& syst_name_);

int main(int argc, char** argv)
{
    __argc__ = argc;
    __argv__ = argv;

    reset_parameters();
    get_user_parameters();
    remind_usage(); // display used parameters

    initialize_objects();
    get_list_of_parameters();
    fill_parameter_nominal_values();

    for( auto &syst_name : __systematics_names_list__){
        std::cout << ALERT << syst_name << " is relative ? " << do_syst_param_has_relative_X_scale(syst_name) << std::endl;
    }

    create_norm_splines();

    //////////////////////////////////////////////////////////////////////
    //  READING GENWEIGHTS FILES
    //////////////////////////////////////////////////////////////////////
    std::map<std::string, std::map<std::string, std::vector<TGraph*>> > graph_list_map;
    for(int i_file = 0 ; i_file < int(__genweights_file_path_list__.size()); i_file++){

        __genweights_file_path__ = __genweights_file_path_list__[i_file];

        std::cout << ALERT << "Openning genWeights file " << __genweights_file_path__ << " " << i_file+1 << "/" << int(__genweights_file_path_list__.size()) << "..." << std::endl;
        if(__genweights_file_path__.empty()){
            std::cout << ERROR << "__genweights_file_path__ not set." << std::endl;
            exit(EXIT_FAILURE);
        }
        if(not GenericToolbox::do_tfile_is_valid(__genweights_file_path__)){
            std::cout << ERROR << __genweights_file_path__ << " can't be opened. Skipping..." << std::endl;
            continue; // skip
        }
        if(__input_genWeights_tfile__ != nullptr){
            __input_genWeights_tfile__->Close();
            delete __input_genWeights_tfile__;
        }
        __input_genWeights_tfile__ = TFile::Open(__genweights_file_path__.c_str(), "READ");
        __sample_sum__ = (TTree*) __input_genWeights_tfile__->Get("sample_sum");
        __flattree__ = (TTree*) __input_genWeights_tfile__->Get("flattree");

        if(__sample_sum__ == nullptr) continue;
        if(__flattree__ == nullptr) continue;

        build_tree_sync_cache();

        std::cout << INFO << "Grabbing Splines Graphs..." << std::endl;
        __sample_sum__->GetEntry(0);
        int nb_of_samples = __sample_sum__->GetLeaf("NSamples")->GetValue(0);
        std::map<std::string, TClonesArray*> clone_array_map;
        GenericToolbox::toggle_quiet_root();
        for( auto &syst_name : __systematic_Xsec_spline_names_list__){

            if(__output_tfile_splines__[syst_name] == nullptr){
                __output_tfile_splines__[syst_name] = TFile::Open(
                    Form("%s_splines.root", syst_name.c_str()),
                    "UPDATE"
                );
            }

            clone_array_map[syst_name] = new TClonesArray("TGraph", nb_of_samples);

            __sample_sum__->SetBranchAddress(
                Form("%sGraph", __from_Xsec_to_genWeights__[syst_name].c_str()),
                &clone_array_map[syst_name]
            );

        }
        GenericToolbox::toggle_quiet_root();

        std::cout << INFO << "Looping over all the genWeights entries..." << std::endl;
        int nb_entries = __sample_sum__->GetEntries();
        int nb_graphs_found = 0;
        for(int i_entry = 0 ; i_entry < nb_entries ; i_entry++){

            GenericToolbox::display_loading(
                i_entry, nb_entries,
                Form("%s%i graphs found", WARNING.c_str(), nb_graphs_found)
            );

            __sample_sum__->GetEntry(i_entry);

            __current_event_bin__ = -1;
            __current_event_sample__   = -1;
            __current_event_reaction__ = -1;
            identify_event_bin(i_entry);

            if(__current_event_bin__ == -1) continue;
            if(not GenericToolbox::does_element_is_in_vector(__current_event_sample__, __samples_list__)) continue;
            if(not GenericToolbox::does_element_is_in_vector(__current_event_reaction__, __reactions_list__)) continue;

            for( auto &syst_name : __systematic_Xsec_spline_names_list__){

                for(int i_sample = 0 ; i_sample < nb_of_samples ; i_sample++ ){

                    if( i_sample >= clone_array_map[syst_name]->GetSize() ) continue;

                    auto* new_graph = (TGraph*)(clone_array_map[syst_name]->At(i_sample));

                    if(
                        clone_array_map[syst_name]->At(i_sample) != nullptr
                        and new_graph->GetN() > 1
                        ){

                        bool splines_are_non_zeroed = false;
                        for(int i_point = 0 ; i_point < new_graph->GetN() ; i_point++){
                            if(new_graph->GetY()[i_point] != 0){
                                splines_are_non_zeroed = true;
                                break;
                            }
                        }
                        if(not splines_are_non_zeroed)
                            continue;

                        std::string bin_name = Form("spline_sam%i_reac%i_bin%i", __current_event_sample__, __current_event_reaction__, __current_event_bin__);

                        new_graph->SetName(bin_name.c_str());
                        nb_graphs_found++;

                        GenericToolbox::toggle_quiet_root(); // ON
                        __output_tfile_splines__[syst_name]->mkdir(Form("Graphs/%s", bin_name.c_str()));
                        GenericToolbox::toggle_quiet_root(); // OFF

                        __output_tfile_splines__[syst_name]->cd(Form("Graphs/%s", bin_name.c_str()));
                        graph_list_map[syst_name][bin_name].emplace_back( (TGraph*) new_graph->Clone() );
                        graph_list_map[syst_name][bin_name].back()->SetMarkerStyle(kFullDotLarge);
                        graph_list_map[syst_name][bin_name].back()->SetMarkerSize(1);
                        graph_list_map[syst_name][bin_name].back()->Write(
                            Form("spline_%i_TGraph",
                                 int(graph_list_map[syst_name][bin_name].size())
                            )
                        );
                        __output_tfile_splines__[syst_name]->cd("");

//            delete new_graph;


                    }
                }

            }

        } // i_entry

        std::cout << INFO << "Freeing up memory..." << std::endl;
        for( auto &syst_name : __systematic_Xsec_spline_names_list__){

            __output_tfile_splines__[syst_name]->Close();
            delete __output_tfile_splines__[syst_name];
            __output_tfile_splines__[syst_name] = nullptr;

            delete clone_array_map[syst_name];

            for( auto &binned_graphs_list : graph_list_map[syst_name] ){
                for( auto& graph : binned_graphs_list.second ){
                    delete graph;
                    graph = nullptr;
                }
                binned_graphs_list.second.clear();
            }
            graph_list_map[syst_name].clear();

        }

    } // i_file


    //////////////////////////////////////////////////////////////////////
    //  AVERAGING GRAPHS
    //////////////////////////////////////////////////////////////////////
    std::cout << INFO << "Averaging Graphs to build Splines" << std::endl;
    TGraph* graph_buffer;
    std::map<std::string, std::map<std::string, TSpline3*> > spline_list_map;
    std::map<std::string, std::map<std::string, TGraph*> > merged_graph_list_map;
    for( auto &syst_name : __systematic_Xsec_spline_names_list__){

        __output_tfile_splines__[syst_name] = TFile::Open(
            Form("%s_splines.root", syst_name.c_str()),
            "READ"
        );

        auto* Graphs_dir = (TDirectory*) __output_tfile_splines__[syst_name]->Get("Graphs");
        if(Graphs_dir == nullptr) continue;
        for(int i_entry = 0 ; i_entry < Graphs_dir->GetListOfKeys()->GetEntries() ; i_entry++){

            std::string bin_name = Graphs_dir->GetListOfKeys()->At(i_entry)->GetName();
            auto* bin_dir = (TDirectory*) __output_tfile_splines__[syst_name]->Get(Form("Graphs/%s", bin_name.c_str()));

            for(int j_entry = 0 ; j_entry < bin_dir->GetListOfKeys()->GetEntries() ; j_entry++){
                graph_list_map[syst_name][bin_name].emplace_back(
                    (TGraph*) __output_tfile_splines__[syst_name]->Get(
                        Form("Graphs/%s/%s", bin_name.c_str(), bin_dir->GetListOfKeys()->At(j_entry)->GetName()))
                );
            }

            if( graph_list_map[syst_name][bin_name].size() >= 1 ){

                std::vector<double> X;
                std::vector<double> Y;
                // look for all X points
                for(auto &graph : graph_list_map[syst_name][bin_name]){
                    for(int i_pt = 0 ; i_pt < graph->GetN() ; i_pt++){
                        if(not GenericToolbox::does_element_is_in_vector(graph->GetX()[i_pt], X)){
                            X.emplace_back(graph->GetX()[i_pt]);
                        }
                    }
                }

                // averaging Y points
                for(int x_index = 0 ; x_index < X.size() ; x_index++){
                    Y.emplace_back(0);
                    int nb_samples = 0;
                    for(auto &graph : graph_list_map[syst_name][bin_name]){
                        for(int i_pt = 0 ; i_pt < graph->GetN() ; i_pt++){
                            if(graph->GetX()[i_pt] == X[x_index]){
                                Y.back() += graph->GetY()[i_pt];
                                nb_samples++;
                            }
                        }
                    }
                    Y.back() /= double(nb_samples);

                    // convert to absolute syst parameter value
                    X[x_index] = from_relative_par_value_to_absolute(syst_name, X[x_index]);
                }

                graph_buffer = new TGraph(X.size(), &X[0], &Y[0]);

                merged_graph_list_map[syst_name][bin_name] = graph_buffer;
                spline_list_map[syst_name][bin_name] =
                    new TSpline3(
                        bin_name.c_str(),
                        graph_buffer
                    );

            } else {
                std::cerr << ALERT << "No graphs found for bin : " << syst_name << ", " << bin_name << std::endl;
            }

        }

        __output_tfile_splines__[syst_name]->Close();

        if(graph_list_map[syst_name].empty()){
            std::cerr << ERROR << "Missing graphs for : " << syst_name << std::endl;
            __missing_splines__.emplace_back(syst_name);
            continue;
        }

        delete __output_tfile_splines__[syst_name];
        __output_tfile_splines__[syst_name] = nullptr;
        for( auto &binned_graphs_list : graph_list_map[syst_name] ){
            for( auto& graph : binned_graphs_list.second ){
                delete graph;
                graph = nullptr;
            }
            binned_graphs_list.second.clear();
        }
        graph_list_map[syst_name].clear();

    }


    //////////////////////////////////////////////////////////////////////
    //  GENERATING MISSING GRAPHS
    //////////////////////////////////////////////////////////////////////
    std::cout << INFO << "Fixing missing graphs" << std::endl;
    for( auto &syst_name : __systematic_Xsec_spline_names_list__) {

        for( int i_sam = 0 ; i_sam < __samples_list__.size() ; i_sam++){
            for( int i_rea = 0 ; i_rea < __reactions_list__.size() ; i_rea++){

                graph_buffer = nullptr;
                // search for a first valid graph
                for( int i_bin = 0 ; i_bin < __bin_edges__.size() ; i_bin++){
                    std::string bin_name = Form("spline_sam%i_reac%i_bin%i", __samples_list__[i_sam], __reactions_list__[i_rea], i_bin);
                    if(merged_graph_list_map[syst_name][bin_name] != nullptr){
                        graph_buffer = merged_graph_list_map[syst_name][bin_name];
                        break;
                    }
                }

                if(graph_buffer != nullptr){ // fill the gaps with nearby graphs
                    for( int i_bin = 0 ; i_bin < __bin_edges__.size() ; i_bin++ ){
                        std::string bin_name = Form("spline_sam%i_reac%i_bin%i", __samples_list__[i_sam], __reactions_list__[i_rea], i_bin);
                        if(merged_graph_list_map[syst_name][bin_name] == nullptr){
                            merged_graph_list_map[syst_name][bin_name] = graph_buffer; // attribute last graph loaded in memory
                        } else {
                            graph_buffer = merged_graph_list_map[syst_name][bin_name];
                        }
                    }
                } else { // create flat splines for empty components

                    std::vector<double> X_points;
                    X_points.emplace_back(-0.75);
                    X_points.emplace_back(-0.5);
                    X_points.emplace_back(-0.25);
                    X_points.emplace_back(0.);
                    X_points.emplace_back(0.25);
                    X_points.emplace_back(0.5);
                    X_points.emplace_back(0.75);
                    for(auto &X_point: X_points){
                        // convert to absolute syst parameter value
                        X_point = from_relative_par_value_to_absolute(syst_name, X_point);
                    }
                    std::vector<double> Y_points;
                    Y_points.emplace_back(1);
                    Y_points.emplace_back(1);
                    Y_points.emplace_back(1);
                    Y_points.emplace_back(1);
                    Y_points.emplace_back(1);
                    Y_points.emplace_back(1);
                    Y_points.emplace_back(1);

                    for( int i_bin = 0 ; i_bin < __bin_edges__.size() ; i_bin++ ){
                        auto* graph = new TGraph(X_points.size(), &X_points[0], &Y_points[0]);
                        std::string bin_name = Form("spline_sam%i_reac%i_bin%i", __samples_list__[i_sam], __reactions_list__[i_rea], i_bin);
                        graph->SetMarkerStyle(kFullDotLarge);
                        graph->SetMarkerSize(1);
                        graph->SetTitle(bin_name.c_str());
                        merged_graph_list_map[syst_name][bin_name] = graph;
                    }

                }

            }
        }

    }


    //////////////////////////////////////////////////////////////////////
    //  SAVING RESULTS
    //////////////////////////////////////////////////////////////////////
    std::cout << INFO << "Writing Splines" << std::endl;
    for(auto &syst_name : __systematic_Xsec_spline_names_list__){

        __output_tfile_splines__[syst_name] = TFile::Open(
            Form("%s_splines.root", syst_name.c_str()),
            "UPDATE"
        );

        __output_tfile_splines__[syst_name]->cd();
        // Write in the ORDER => in the fitter its needed
        for( int i_sam = 0 ; i_sam < __samples_list__.size() ; i_sam++){
            for( int i_rea = 0 ; i_rea < __reactions_list__.size() ; i_rea++){
                for( int i_bin = 0 ; i_bin < __bin_edges__.size() ; i_bin++ ){
                    std::string bin_name = Form("spline_sam%i_reac%i_bin%i", __samples_list__[i_sam], __reactions_list__[i_rea], i_bin);
                    merged_graph_list_map[syst_name][bin_name]->SetMarkerStyle(kFullDotLarge);
                    merged_graph_list_map[syst_name][bin_name]->SetMarkerSize(1);
                    merged_graph_list_map[syst_name][bin_name]->SetTitle(bin_name.c_str());
                    merged_graph_list_map[syst_name][bin_name]->Write(bin_name.c_str());
                }
            }
        }

        __output_tfile_splines__[syst_name]->mkdir("Splines");
        __output_tfile_splines__[syst_name]->cd("Splines");
        for(auto &spline : spline_list_map[syst_name]){

            spline.second->SetLineWidth(1);
            spline.second->SetTitle(spline.first.c_str());
            spline.second->Write(spline.first.c_str());

        }

        __output_tfile_splines__[syst_name]->Close();

    }


    //////////////////////////////////////////////////////////////////////
    //  CHECKING SPLINES AT NOMINAL VALUE
    //////////////////////////////////////////////////////////////////////
    std::cout << INFO << "Checking written splines at nominal value..." << std::endl;
    for(auto &syst_name : __systematics_names_list__){

        if(GenericToolbox::does_element_is_in_vector(syst_name, __missing_splines__)){
            continue;
        }

        __output_tfile_splines__[syst_name] = TFile::Open(
            Form("%s_splines.root", syst_name.c_str()),
            "READ"
        );

        __output_tfile_splines__[syst_name]->cd();
        // Write in the ORDER => in the fitter its needed
        TGraph* graph_buffer=nullptr;
        TSpline3* spline_buffer=nullptr;
        for( int i_sam = 0 ; i_sam < __samples_list__.size() ; i_sam++){
            for( int i_rea = 0 ; i_rea < __reactions_list__.size() ; i_rea++){
                for( int i_bin = 0 ; i_bin < __bin_edges__.size() ; i_bin++ ){
                    std::string bin_name = Form("spline_sam%i_reac%i_bin%i", __samples_list__[i_sam], __reactions_list__[i_rea], i_bin);
                    graph_buffer = (TGraph*) __output_tfile_splines__[syst_name]->Get(bin_name.c_str());
                    spline_buffer = new TSpline3(bin_name.c_str(), graph_buffer);
                    double weight = spline_buffer->Eval(__parameter_nominal_values_map__[syst_name]);
                    if( weight!=weight or fabs(weight-1) > 0.001){
                        std::cout << ERROR << "CAUTION: Spline " << syst_name << "(sample=" << i_sam;
                        std::cout << ", reaction=" << i_rea << ", bin=" << i_bin;
                        std::cout << ") has weight=" << weight << " at nominal value." << std::endl;
                    }
                    delete spline_buffer;
                }
            }
        }

        __output_tfile_splines__[syst_name]->Close();
    }

    //////////////////////////////////////////////////////////////////////
    //  REMOVING FILES FOR MISSING SYSTEMATICS
    //////////////////////////////////////////////////////////////////////
    std::cout << INFO << "Removing Missing Splines Files" << std::endl;
    for(auto const &missing_spline_name : __missing_splines__){
        std::remove(Form("%s_splines.root", missing_spline_name.c_str()));
    }
    if(not __missing_splines__.empty()){
        regenarte_covariance_matrix_file();
    }


    //////////////////////////////////////////////////////////////////////
    //  GENERATING CONFIGS FOR THE FITTER
    //////////////////////////////////////////////////////////////////////
    generate_json_config_file();
    destroy_objects();

    exit(EXIT_SUCCESS);

}


std::string remind_usage(){

    std::stringstream remind_usage_ss;
    remind_usage_ss << "*********************************************************" << std::endl;
    remind_usage_ss << " > Command Line Arguments" << std::endl;
    remind_usage_ss << "  -w : genweights input file (Current : " << __genweights_file_path__ << ")" << std::endl;
    remind_usage_ss << "  -l : genweights input file list (Current : " << __genweights_file_list_path__ << ")" << std::endl;
    remind_usage_ss << "  -b : binning file (Current : " << __binning_file_path__ << ")" << std::endl;
    remind_usage_ss << "  -t : tree converter file (Current : " << __tree_converter_file_path__ << ")" << std::endl;
    remind_usage_ss << "  -c : file containing infos on the covariance matrix (Current : " << __covariance_matrix_file_path__ << ")" << std::endl;
    remind_usage_ss << "*********************************************************" << std::endl;

    std::cerr << remind_usage_ss.str();
    return remind_usage_ss.str();

}
void reset_parameters(){
    __json_config_path__ = "";
    __genweights_file_path__ = "";
    __genweights_file_list_path__ = "";
    __binning_file_path__ = "";
    __genweights_file_path_list__.clear();
}
void get_user_parameters(){

    if(__argc__ == 1){
        remind_usage();
        exit(EXIT_FAILURE);
    }

    std::cout << WARNING << "Sanity check" << std::endl;

    const std::string XSLLHFITTER = std::getenv("XSLLHFITTER");
    if(XSLLHFITTER.empty()){

        std::cerr << ERROR << "Environment variable \"XSLLHFITTER\" not set." << std::endl
                  << ERROR << "Cannot determine source tree location." << std::endl;
        remind_usage();
        exit(EXIT_FAILURE);
    }

    std::cout << WARNING << "Reading user parameters" << std::endl;

    for(int i_arg = 0; i_arg < __argc__; i_arg++){
        __command_line__ += __argv__[i_arg];
        __command_line__ += " ";
    }

    for(int i_arg = 0; i_arg < __argc__; i_arg++){

        if(std::string(__argv__[i_arg]) == "-j"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __json_config_path__ = std::string(__argv__[j_arg]);
                if(not GenericToolbox::do_path_is_file(__json_config_path__)){
                    std::cerr << ERROR << std::string(__argv__[i_arg]) << ": " << __json_config_path__ << " could not be found." << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                std::cout << ERROR << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-w"){
            if (i_arg < __argc__ - 1) {
                do{
                    int j_arg = i_arg + 1;
                    if(not GenericToolbox::do_path_is_file(std::string(__argv__[j_arg]))){
                        std::cerr << ERROR << std::string(__argv__[i_arg]) << ": " << std::string(__argv__[j_arg]) << " could not be found." << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    __genweights_file_path_list__.emplace_back(std::string(__argv__[j_arg]));
                    if(__argv__[j_arg+1][0] == '-') break;
                    i_arg++; // changing i_arg for next loop
                } while(i_arg < __argc__ - 1);
            } else {
                std::cout << ERROR << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-b"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __binning_file_path__ = std::string(__argv__[j_arg]);
                if(not GenericToolbox::do_path_is_file(__binning_file_path__)){
                    std::cerr << ERROR << std::string(__argv__[i_arg]) << ": " << __binning_file_path__ << " could not be found." << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                std::cout << ERROR << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-t"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __tree_converter_file_path__ = std::string(__argv__[j_arg]);
                if(not GenericToolbox::do_path_is_file(__tree_converter_file_path__)){
                    std::cerr << ERROR << std::string(__argv__[i_arg]) << ": " << __covariance_matrix_file_path__ << " could not be found." << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                std::cout << ERROR << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-c"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __covariance_matrix_file_path__ = std::string(__argv__[j_arg]);
                if(not GenericToolbox::do_path_is_file(__covariance_matrix_file_path__)){
                    std::cerr << ERROR << std::string(__argv__[i_arg]) << ": " << __covariance_matrix_file_path__ << " could not be found." << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                std::cout << ERROR << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-l"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __genweights_file_list_path__ = std::string(__argv__[j_arg]);
                if(not GenericToolbox::do_path_is_file(__genweights_file_list_path__)){
                    std::cerr << ERROR << std::string(__argv__[i_arg]) << ": " << __genweights_file_list_path__ << " could not be found." << std::endl;
                    exit(EXIT_FAILURE);
                }
                read_genweights_files_list();
            } else {
                std::cout << ERROR << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }

    }

}
void initialize_objects(){

    std::cout << INFO << "Building component name mapping from Xsec to GenWeights conventions..." << std::endl;
    fill_component_mapping();

    std::cout << INFO << "Defining the binning..." << std::endl;
    if(__binning_file_path__.empty()){
        std::cout << ERROR << "__binning_file_path__ not set." << std::endl;
        exit(EXIT_FAILURE);
    }
    define_binning(__binning_file_path__);

    std::cout << INFO << "Openning TreeConverter file..." << std::endl;
    if(__tree_converter_file_path__.empty()){
        std::cout << ERROR << "__tree_converter_file_path__ not set." << std::endl;
        exit(EXIT_FAILURE);
    }
    __tree_converter_tfile__ = TFile::Open(__tree_converter_file_path__.c_str(), "READ");
    __tree_converter_reco_ttree__ = (TTree*) __tree_converter_tfile__->Get("selectedEvents");

}
void destroy_objects(){

    if(__input_genWeights_tfile__ != nullptr) __input_genWeights_tfile__->Close();
    if(__tree_converter_tfile__ != nullptr) __tree_converter_tfile__->Close();
    __genweights_file_path_list__.clear();

}


void get_list_of_parameters(){

    std::cout << WARNING << "Getting list of parameters..." << std::endl;

    if(__covariance_matrix_file_path__.empty()){
        std::cerr << ERROR << "__covariance_matrix_file_path__ is not set." << std::endl;
        exit(EXIT_FAILURE);
    }

    TFile* covariance_matrix_tfile = TFile::Open(__covariance_matrix_file_path__.c_str(), "READ");
    if(not covariance_matrix_tfile->IsOpen()){
        std::cerr << ERROR << "Could not open : " << covariance_matrix_tfile->GetName() << std::endl;
        exit(EXIT_FAILURE);
    }

    auto* xsec_param_names = dynamic_cast<TObjArray *>(covariance_matrix_tfile->Get("xsec_param_names"));
    for(int i_parameter = 0 ; i_parameter < xsec_param_names->GetEntries() ; i_parameter++){

        __systematics_names_list__.emplace_back(xsec_param_names->At(i_parameter)->GetName());

        if(not __from_Xsec_to_genWeights__[xsec_param_names->At(i_parameter)->GetName()].empty()){
            std::cout << INFO << "  -> GenWeights component : \"" << __from_Xsec_to_genWeights__[xsec_param_names->At(i_parameter)->GetName()];
            std::cout << "\" has been identified as " << xsec_param_names->At(i_parameter)->GetName() << "." << std::endl;
        } else {
            std::cout << ALERT << "  -> No equivalent for " << xsec_param_names->At(i_parameter)->GetName() << " in GenWeights. Will be treated as a norm factor." << std::endl;
            __systematics_norm_splines_names_list__.emplace_back(xsec_param_names->At(i_parameter)->GetName());
//            __missing_splines__.emplace_back(xsec_param_names->At(i_parameter)->GetName());
            continue;
        }

        __systematic_Xsec_spline_names_list__.emplace_back(xsec_param_names->At(i_parameter)->GetName());

        __output_tfile_splines__[xsec_param_names->At(i_parameter)->GetName()] = TFile::Open(
            Form("%s_splines.root", xsec_param_names->At(i_parameter)->GetName()),
            "RECREATE"
        );
        __output_tfile_splines__[xsec_param_names->At(i_parameter)->GetName()]->mkdir("Graphs"); // where graphs will be stored

    }

    covariance_matrix_tfile->Close();

}
void create_norm_splines(){

    std::cout << WARNING << "Creating norm splines..." << std::endl;

    if(__covariance_matrix_file_path__.empty()){
        std::cerr << ERROR << "__covariance_matrix_file_path__ is not set." << std::endl;
        exit(EXIT_FAILURE);
    }

    TFile* covariance_matrix_tfile = TFile::Open(__covariance_matrix_file_path__.c_str(), "READ");
    if(not covariance_matrix_tfile->IsOpen()){
        std::cerr << ERROR << "Could not open : " << covariance_matrix_tfile->GetName() << std::endl;
        exit(EXIT_FAILURE);
    }
    TObjArray* xsec_param_names = dynamic_cast<TObjArray *>(covariance_matrix_tfile->Get("xsec_param_names"));
    TVectorT<double>* xsec_param_lb = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_lb"));
    TVectorT<double>* xsec_param_ub = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_ub"));

    int nb_samples = 8;
    for(int i_sam = 0 ; i_sam < nb_samples; i_sam++)
        __samples_list__.emplace_back(i_sam);
    int nb_reactions = 10;
    for(int i_rea = 0 ; i_rea < nb_reactions ; i_rea++) __reactions_list__.emplace_back(i_rea);
    // check for extra cases
    __tree_converter_reco_ttree__->SetBranchStatus("*", false); // disable all branches -> faster GetEntry()
    __tree_converter_reco_ttree__->SetBranchStatus("reaction", true);
    __tree_converter_reco_ttree__->SetBranchStatus("cut_branch", true);

    for( int i_entry = 0 ; i_entry < __tree_converter_reco_ttree__->GetEntries() ; i_entry++){
        GenericToolbox::display_loading(i_entry, __tree_converter_reco_ttree__->GetEntries(), WARNING + "Looking for the samples and reactions...");
        __tree_converter_reco_ttree__->GetEntry(i_entry);

//        __current_event_sample__ = int(__tree_converter_reco_ttree__->GetLeaf("sample")->GetValue());
        __current_event_sample__ = int(__tree_converter_reco_ttree__->GetLeaf("cut_branch")->GetValue());
        __current_event_reaction__ = int(__tree_converter_reco_ttree__->GetLeaf("reaction")->GetValue());
        if(__current_event_sample__ == -1) continue;
        if(__current_event_reaction__ == -1 or __current_event_reaction__ == 999) continue;

        if(not GenericToolbox::does_element_is_in_vector(
               int(__tree_converter_reco_ttree__->GetLeaf("reaction")->GetValue()),
               __reactions_list__)
           ){
            __reactions_list__.emplace_back(int(__tree_converter_reco_ttree__->GetLeaf("reaction")->GetValue()));
        }
        if(not GenericToolbox::does_element_is_in_vector(
//            int(__tree_converter_reco_ttree__->GetLeaf("sample")->GetValue()),
            int(__tree_converter_reco_ttree__->GetLeaf("cut_branch")->GetValue()),
               __samples_list__)
            ){
//            __samples_list__.emplace_back(int(__tree_converter_reco_ttree__->GetLeaf("sample")->GetValue()));
            __samples_list__.emplace_back(int(__tree_converter_reco_ttree__->GetLeaf("cut_branch")->GetValue()));
        }
    }
    __tree_converter_reco_ttree__->SetBranchStatus("*", true);

    std::cout << INFO << "Now writing norm splines..." << std::endl;
    for(auto const& norm_spline_name : __systematics_norm_splines_names_list__){

        std::vector<double> X_points;
        X_points.emplace_back(-0.75);
        X_points.emplace_back(-0.5);
        X_points.emplace_back(-0.25);
        X_points.emplace_back(0.);
        X_points.emplace_back(0.25);
        X_points.emplace_back(0.5);
        X_points.emplace_back(0.75);
        std::vector<double> Y_points = X_points;
        for(auto& Y_point : Y_points) Y_point += 1.;
        std::vector<double> Y_flat_points(X_points.size());
        for(auto& Y_flat_point : Y_flat_points) Y_flat_point = 1;
        for(auto &X_point: X_points){
            // convert to absolute syst parameter value
            X_point = from_relative_par_value_to_absolute(norm_spline_name, X_point);
        }

        __output_tfile_splines__[norm_spline_name] = TFile::Open(
            Form("%s_splines.root", norm_spline_name.c_str()), "RECREATE");
        __output_tfile_splines__[norm_spline_name]->mkdir("Splines");

        std::vector<int> valid_reaction_index_list;
        std::vector<int> exclude_sample_index_list;
        // https://t2k.org/comm/pubboard/review/OA2020/technical-notes/niwg/TN344/TN344
        if(GenericToolbox::do_string_starts_with_substring(norm_spline_name, "2p2h_norm")){
            // 2p2h_norm* - 2p2h
            valid_reaction_index_list.emplace_back(__reaction_name_to_index__["2p2h"]);
        }
        else if(GenericToolbox::do_string_starts_with_substring(norm_spline_name, "Q2_")){
            // Q2_* - CCQE
            valid_reaction_index_list.emplace_back(__reaction_name_to_index__["CCQE"]);
        }
        else if(GenericToolbox::do_string_starts_with_substring(norm_spline_name, "EB_")){
            // EB* - CCQE
            valid_reaction_index_list.emplace_back(__reaction_name_to_index__["CCQE"]);
        }
        else if(GenericToolbox::do_string_starts_with_substring(norm_spline_name, "CC_norm_")){
            // CC_norm_nu - All nu CC
            // CC_norm_anu - All nubar CC
            valid_reaction_index_list.emplace_back(__reaction_name_to_index__["CCQE"]);
        }
        else if(GenericToolbox::do_string_starts_with_substring(norm_spline_name, "nue_")){
            // nue_numu - All nue
            // nuebar_numubar - All nuebar
            for(auto const& reaction : __reaction_name_to_index__){
                valid_reaction_index_list.emplace_back(reaction.second);
            }
        }
        else if(GenericToolbox::do_string_starts_with_substring(norm_spline_name, "CC_Misc")){
            // CC_Misc - CCGamma, CCKaon, CCEta
            // CC Misc Misc Spline 100% normalisation error on CC1γ, CC1K, CC1η.
            // all reactions but -> only CC-Other
            exclude_sample_index_list.emplace_back(__samples_name_to_index__["CC-0pi"]);
            exclude_sample_index_list.emplace_back(__samples_name_to_index__["CC-1pi"]);
            for(auto const& reaction : __reaction_name_to_index__){
                valid_reaction_index_list.emplace_back(reaction.second);
            }
        }
        else if(GenericToolbox::do_string_starts_with_substring(norm_spline_name, "CC_DIS_MultPi")){
            // CC_DIS_MultiPi_Norm_Nu - CCDIS and CCMultPi, nu only
            // CC_DIS_MultiPi_Norm_nuBar - CCDIS and CCMultPi, anu only
            exclude_sample_index_list.emplace_back(__samples_name_to_index__["CC-0pi"]);
            exclude_sample_index_list.emplace_back(__samples_name_to_index__["CC-1pi"]);
            valid_reaction_index_list.emplace_back(__reaction_name_to_index__["DIS"]);
        }
        else if(GenericToolbox::do_string_starts_with_substring(norm_spline_name, "CC_Coh_")){
            //
            exclude_sample_index_list.emplace_back(__samples_name_to_index__["CC-0pi"]);
            exclude_sample_index_list.emplace_back(__samples_name_to_index__["CC-1pi"]);
            valid_reaction_index_list.emplace_back(__reaction_name_to_index__["Coh"]);
        }
        else if(GenericToolbox::do_string_starts_with_substring(norm_spline_name, "NC_other_near")){
            // NC_other_near - all NC which isn't Coh or 1gamma
            for(auto const& reaction : __reaction_name_to_index__){
                valid_reaction_index_list.emplace_back(reaction.second);
            }
        }

        for(auto const& sample_id : __samples_list__){
            for(auto const& reaction_id : __reactions_list__){
                for(int i_bin = 0 ; i_bin < int(__bin_edges__.size()) ; i_bin++){

                    TGraph* graph = nullptr;

                    __output_tfile_splines__[norm_spline_name]->cd();
                    if(not GenericToolbox::does_element_is_in_vector(sample_id,exclude_sample_index_list)
                       and GenericToolbox::does_element_is_in_vector(reaction_id,valid_reaction_index_list)
                       ){
                        graph = new TGraph(X_points.size(), &X_points[0], &Y_points[0]);
                    }
                    else{
                        graph = new TGraph(X_points.size(), &X_points[0], &Y_flat_points[0]);
                    }

                    std::string bin_name = Form("spline_sam%i_reac%i_bin%i", sample_id, reaction_id, i_bin);
                    graph->SetMarkerStyle(kFullDotLarge);
                    graph->SetMarkerSize(1);
                    graph->SetTitle(bin_name.c_str());
                    graph->Write(bin_name.c_str());

                    __output_tfile_splines__[norm_spline_name]->cd("Splines");
                    auto* spline = new TSpline3(bin_name.c_str(),graph);
                    spline->SetLineWidth(1);
                    spline->SetTitle(bin_name.c_str());
                    spline->Write(bin_name.c_str());

                }
            }
        }

        __output_tfile_splines__[norm_spline_name]->Close();

    }
    std::cout << INFO << "Norm splines have been written." << std::endl;



}
void regenarte_covariance_matrix_file(){

    std::cout << INFO << "Regenerating the covariance matrix file with only valid components..." << std::endl;

    if(__covariance_matrix_file_path__.empty()){
        std::cerr << ERROR << "__covariance_matrix_file_path__ is not set." << std::endl;
        exit(EXIT_FAILURE);
    }

    TFile* covariance_matrix_tfile = TFile::Open(__covariance_matrix_file_path__.c_str(), "READ");
    if(not covariance_matrix_tfile->IsOpen()){
        std::cerr << ERROR << "Could not open : " << covariance_matrix_tfile->GetName() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::map<std::string, TObjArray*> TObjArray_list;
    TObjArray_list["xsec_param_names"] = dynamic_cast<TObjArray *>(covariance_matrix_tfile->Get("xsec_param_names"));
//    TObjArray_list["xsec_norm_modes"] = dynamic_cast<TObjArray *>(covariance_matrix_tfile->Get("xsec_norm_modes"));
//    TObjArray_list["xsec_norm_elements"] = dynamic_cast<TObjArray *>(covariance_matrix_tfile->Get("xsec_norm_elements"));
//    TObjArray_list["xsec_norm_nupdg"] = dynamic_cast<TObjArray *>(covariance_matrix_tfile->Get("xsec_norm_nupdg"));

    std::map<std::string, TVectorT<double>*> TVectorT_double_list;
    TVectorT_double_list["xsec_param_nom"] = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_nom"));
    TVectorT_double_list["xsec_param_nom_unnorm"] = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_nom_unnorm"));
    TVectorT_double_list["xsec_param_prior"] = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_prior"));
    TVectorT_double_list["xsec_param_prior_unnorm"] = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_prior_unnorm"));
    TVectorT_double_list["xsec_param_lb"] = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_lb"));
    TVectorT_double_list["xsec_param_ub"] = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_ub"));

    std::map<std::string, TMatrixT<double>*> TMatrixT_double_list;
    TMatrixT_double_list["xsec_param_id"] = (TMatrixT<double> *)(covariance_matrix_tfile->Get("xsec_param_id"));
    TMatrixT_double_list["xsec_cov"] = (TMatrixT<double> *)(covariance_matrix_tfile->Get("xsec_cov"));

    auto* hcov = (TH2D *)(covariance_matrix_tfile->Get("hcov"));


    // declaring new objects
    TFile *chopped_covariance_matrix_tfile = TFile::Open("chopped_covariance_matrix.root", "RECREATE");
    chopped_covariance_matrix_tfile->cd();
    std::map<std::string, TObjArray*> chopped_TObjArray_list;
    for(auto const& original_TObjArray : TObjArray_list){
        chopped_TObjArray_list[original_TObjArray.first] = new TObjArray(original_TObjArray.second->GetSize() - int(__missing_splines__.size()));
    }
    std::map<std::string, TVectorT<double>*> chopped_TVectorT_double_list;
    for(auto const& original_TVectorT_double : TVectorT_double_list){
        chopped_TVectorT_double_list[original_TVectorT_double.first] = new TVectorT<double>(original_TVectorT_double.second->GetNrows() - int(__missing_splines__.size()));
    }
    std::map<std::string, TMatrixT<double>*> chopped_TMatrixT_double_list;
    for(auto const& original_TMatrixT_double : TMatrixT_double_list){
        int nb_rows = original_TMatrixT_double.second->GetNrows() - int(__missing_splines__.size());
        int nb_cols = original_TMatrixT_double.second->GetNcols() - int(__missing_splines__.size());
        if(original_TMatrixT_double.second->GetNcols() != original_TMatrixT_double.second->GetNrows()){
            nb_cols = original_TMatrixT_double.second->GetNcols();
        }
        chopped_TMatrixT_double_list[original_TMatrixT_double.first] = new TMatrixT<double>(nb_rows, nb_cols);
    }
    auto* chopped_hcov = new TH2D(hcov->GetName(), hcov->GetTitle(),
                                  hcov->GetNbinsX() - int(__missing_splines__.size()), hcov->GetXaxis()->GetXmin(), hcov->GetXaxis()->GetXmax() - __missing_splines__.size(),
                                  hcov->GetNbinsY() - int(__missing_splines__.size()), hcov->GetYaxis()->GetXmin(), hcov->GetYaxis()->GetXmax() - __missing_splines__.size());

    // copying data
    int i_chopped_parameter = 0;
    for(int i_parameter = 0 ; i_parameter < TObjArray_list["xsec_param_names"]->GetEntries() ; i_parameter++){

        if(GenericToolbox::does_element_is_in_vector(TObjArray_list["xsec_param_names"]->At(i_parameter)->GetName(),__missing_splines__)){
            continue;
        }

        for(auto const& original_TObjArray : TObjArray_list){
            chopped_TObjArray_list[original_TObjArray.first]->Add(original_TObjArray.second->At(i_parameter));
        }

        for(auto const& original_TVectorT_double : TVectorT_double_list){
            (*chopped_TVectorT_double_list[original_TVectorT_double.first])[i_chopped_parameter] =
                (*original_TVectorT_double.second)[i_parameter];
        }

        int j_chopped_parameter = 0;
        for(int j_parameter = 0 ; j_parameter < TObjArray_list["xsec_param_names"]->GetEntries() ; j_parameter++){

            if(GenericToolbox::does_element_is_in_vector(TObjArray_list["xsec_param_names"]->At(j_parameter)->GetName(),__missing_splines__)){
                continue;
            }

            for(auto const& original_TMatrixT_double : TMatrixT_double_list){
                if(original_TMatrixT_double.second->GetNrows() == original_TMatrixT_double.second->GetNcols()){ // if squared matrix
                    (*chopped_TMatrixT_double_list[original_TMatrixT_double.first])[i_chopped_parameter][j_chopped_parameter] =
                        (*original_TMatrixT_double.second)[i_parameter][j_parameter];
                } else {
                    if(j_parameter < original_TMatrixT_double.second->GetNcols()){
                        (*chopped_TMatrixT_double_list[original_TMatrixT_double.first])[i_chopped_parameter][j_parameter] =
                            (*original_TMatrixT_double.second)[i_parameter][j_parameter];
                    }
                }

            }

            chopped_hcov->SetBinContent(i_chopped_parameter+1, j_chopped_parameter+1,
                                        hcov->GetBinContent(i_parameter+1, j_parameter+1)
            );
            chopped_hcov->GetXaxis()->SetBinLabel(i_chopped_parameter+1, TObjArray_list["xsec_param_names"]->At(i_parameter)->GetName());
            chopped_hcov->GetYaxis()->SetBinLabel(j_chopped_parameter+1, TObjArray_list["xsec_param_names"]->At(j_parameter)->GetName());

            j_chopped_parameter++;
        }
        i_chopped_parameter++;

    }
    chopped_hcov->SetMinimum(hcov->GetMinimum());
    chopped_hcov->SetMaximum(hcov->GetMaximum());
    chopped_hcov->GetXaxis()->SetBit(TAxis::kLabelsVert);
    chopped_hcov->GetYaxis()->SetBit(TAxis::kLabelsVert);
    chopped_hcov->GetZaxis()->SetTitle(hcov->GetZaxis()->GetTitle());

    // writing data
    for(auto const& chopped_TObjArray : chopped_TObjArray_list){
        chopped_covariance_matrix_tfile->mkdir(chopped_TObjArray.first.c_str());
        chopped_covariance_matrix_tfile->cd(chopped_TObjArray.first.c_str());
        chopped_TObjArray.second->Write();
        chopped_covariance_matrix_tfile->cd("");
    }
    for(auto const& chopped_TVectorT_double : chopped_TVectorT_double_list){
        chopped_TVectorT_double.second->Write(chopped_TVectorT_double.first.c_str());
    }
    for(auto const& chopped_TMatrixT_double : chopped_TMatrixT_double_list){
        chopped_TMatrixT_double.second->Write(chopped_TMatrixT_double.first.c_str());
    }
    chopped_hcov->Write();
    chopped_covariance_matrix_tfile->Close();
    covariance_matrix_tfile->Close();

}
void generate_json_config_file(){

    std::cout << WARNING << "Generating json config file..." << std::endl;

    if(__covariance_matrix_file_path__.empty()){
        std::cerr << ERROR << "__covariance_matrix_file_path__ is not set." << std::endl;
        exit(EXIT_FAILURE);
    }

    TFile* covariance_matrix_tfile = TFile::Open(__covariance_matrix_file_path__.c_str(), "READ");
    if(not covariance_matrix_tfile->IsOpen()){
        std::cerr << ERROR << "Could not open : " << covariance_matrix_tfile->GetName() << std::endl;
        exit(EXIT_FAILURE);
    }

    TObjArray* xsec_param_names = dynamic_cast<TObjArray *>(covariance_matrix_tfile->Get("xsec_param_names"));
    TVectorT<double>* xsec_param_nom_unnorm = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_nom_unnorm"));
//  TVectorT<double>* xsec_param_prior_unnorm = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_prior_unnorm"));
    TVectorT<double>* xsec_param_lb = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_lb"));
    TVectorT<double>* xsec_param_ub = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_ub"));

    std::vector<std::string> dial_strings;
    for(int i_parameter = 0 ; i_parameter < xsec_param_names->GetEntries() ; i_parameter++){

        if(
            GenericToolbox::does_element_is_in_vector(
                xsec_param_names->At(i_parameter)->GetName(),
                __missing_splines__)){
            continue;
        }

        std::stringstream dial_ss;
        dial_ss << "    {" << std::endl;
        dial_ss << "      \"name\" : \"" << xsec_param_names->At(i_parameter)->GetName() << "\"," << std::endl;
        dial_ss << "      \"nominal\" : " << (*xsec_param_nom_unnorm)[i_parameter] << "," << std::endl;
        dial_ss << "      \"step\" : 0.05," << std::endl;
        dial_ss << "      \"limit_lo\" : " << (*xsec_param_lb)[i_parameter] << "," << std::endl;
        dial_ss << "      \"limit_hi\" : " << (*xsec_param_ub)[i_parameter] << "," << std::endl;
        dial_ss << "      \"binning\" : \"" << __binning_file_path__ << "\"," << std::endl;
        dial_ss << "      \"splines\" : \"" << __output_tfile_splines__[xsec_param_names->At(i_parameter)->GetName()]->GetName() << "\"," << std::endl;
        dial_ss << "      \"use\" : true" << std::endl;
        dial_ss << "    }";

        dial_strings.emplace_back(dial_ss.str());

    }

    std::vector<std::string> input_dir_elements;
    input_dir_elements.emplace_back("inputs");
    input_dir_elements.emplace_back("BANFF_Fit");
    auto cwd = GenericToolbox::get_current_working_folder_path();
    auto path_elements = GenericToolbox::split_string(cwd, "/");
    bool is_triggered = false;
    for(int i_folder = 0 ; i_folder < int(path_elements.size()) ; i_folder++){
        if(is_triggered){
            input_dir_elements.emplace_back(path_elements[i_folder]);
        }
        else if(path_elements[i_folder] == input_dir_elements[1]){
            is_triggered = true;
        }
    }

    std::stringstream config_ss;
    config_ss << "{" << std::endl;
    config_ss << "  \"input_dir\" : \"/" << GenericToolbox::join_vector_string(input_dir_elements, "/") << "/\"," << std::endl;
    config_ss << "  \"dimensions\" : [" << __reactions_list__.size()*__bin_edges__.size() << ", " << __bin_edges__.size() << "]," << std::endl;
    config_ss << "  \"dials\" : " << std::endl;
    config_ss << "  [" << std::endl;
    config_ss << GenericToolbox::join_vector_string(dial_strings, ",\n") << std::endl;
    config_ss << "  ]" << std::endl;
    config_ss << "}" << std::endl;
    GenericToolbox::write_string_in_file("./config_splines.json", config_ss.str());

    covariance_matrix_tfile->Close();

}
void read_genweights_files_list(){

    __genweights_file_path_list__ = GenericToolbox::read_file(__genweights_file_list_path__);

}
void define_binning(std::string binning_file_) {

    std::ifstream fin(binning_file_, std::ios::in);
    if(!fin.is_open())
    {
        std::cerr << ERROR << "Can't open binning file." << std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        std::string line;
        while(std::getline(fin, line))
        {
            std::stringstream ss(line);
            double D1_1, D1_2, D2_1, D2_2;
            if(!(ss >> D2_1 >> D2_2 >> D1_1 >> D1_2))
            {
                std::cerr << WARNING << "Bad line format: " << line << std::endl;
                continue;
            }
            __bin_edges__.emplace_back(xsllh::FitBin(D1_1, D1_2, D2_1, D2_2));
//      std::cout << INFO << D1_1 << "\t" << D1_2 << "\t" << D2_1 << "\t" << D2_2 << std::endl;
        }
    }
    fin.close();

}
int identify_event_bin(int entry_) {

    __flattree__->SetBranchStatus("*", false);
    __flattree__->SetBranchStatus("sTrueVertexID", true);
    __flattree__->SetBranchStatus("sTrueVertexNuEnergy", true);
    __tree_converter_reco_ttree__->SetBranchStatus("*", false);
//    __tree_converter_reco_ttree__->SetBranchStatus("sample", true);
    __tree_converter_reco_ttree__->SetBranchStatus("cut_branch", true);
    __tree_converter_reco_ttree__->SetBranchStatus("reaction", true);
    __tree_converter_reco_ttree__->SetBranchStatus("D1Reco", true);
    __tree_converter_reco_ttree__->SetBranchStatus("D2Reco", true);
    if(__tree_sync_cache__[entry_] != -1){
        __tree_converter_reco_ttree__->GetEntry(__tree_sync_cache__[entry_]);

        __current_event_sample__
//            = int(__tree_converter_reco_ttree__->GetLeaf("sample")->GetValue(0));
            = int(__tree_converter_reco_ttree__->GetLeaf("cut_branch")->GetValue(0));
        __current_event_reaction__ = int(__tree_converter_reco_ttree__->GetLeaf("reaction")->GetValue(0));

        double D1 = __tree_converter_reco_ttree__->GetLeaf("D1Reco")->GetValue(0);
        double D2 = __tree_converter_reco_ttree__->GetLeaf("D2Reco")->GetValue(0);

        bool bin_has_been_found = false;
        for(int i_bin = 0 ; i_bin < __bin_edges__.size() ; i_bin++){
            if(
                D1 >= __bin_edges__[i_bin].D1low
                and D1 < __bin_edges__[i_bin].D1high
                and D2 >= __bin_edges__[i_bin].D2low
                and D2 < __bin_edges__[i_bin].D2high
                ){
                __current_event_bin__ = i_bin;
                bin_has_been_found = true;
                break;
            }
        }
        if(not bin_has_been_found){
            __flattree__->GetEntry(entry_); // flat tree is synchronized with sample_sum
            int vertexID = __flattree__->GetLeaf("sTrueVertexID")->GetValue(0);
            double nu_energy = __flattree__->GetLeaf("sTrueVertexNuEnergy")->GetValue(0);
            std::cerr << "BIN NOT FOUND for D1=" << D1 << " and D2=" << D2 << " : vertexID=" << vertexID << " / nu_energy=" << nu_energy << std::endl;
        }
    }
    __tree_converter_reco_ttree__->SetBranchStatus("*", true);
    __flattree__->SetBranchStatus("*", true);

    return -1;

}

void fill_names_list(){

    __samples_name_to_index__["CC-0pi"] = 0;
    __samples_name_to_index__["CC-1pi"] = 1;
    __samples_name_to_index__["CC-Other"] = 2;

    __reaction_name_to_index__["CCQE"] = 0;
    __reaction_name_to_index__["2p2h"] = 9;
    __reaction_name_to_index__["RES"] = 1;
    __reaction_name_to_index__["DIS"] = 2;
    __reaction_name_to_index__["COH"] = 3;
    __reaction_name_to_index__["NC"] = 4;
    __reaction_name_to_index__["CC-#bar{#nu}_{#mu}"] = 5;
    __reaction_name_to_index__["CC-#nu_{e}, CC-#bar{#nu}_{e}"] = 6;
    __reaction_name_to_index__["out FV"] = 7;
    __reaction_name_to_index__["other"] = 999;
    __reaction_name_to_index__["no truth"] = -1;
    __reaction_name_to_index__["sand #mu"] = 777;

}

void build_tree_sync_cache(){

    map_tree_converter_entries();

    std::cerr << INFO << "Building tree sync cache..." << std::endl;
    int nb_entries = __sample_sum__->GetEntries(); // genWeights tree (same number of events as __flattree__)

    Int_t sTrueVertexID;
    Int_t sRun;
    Int_t vertexID;

    __tree_sync_cache__.clear();

    int nb_matches = 0, nb_missing = 0;
    __flattree__->SetBranchStatus("*", false);
    __flattree__->SetBranchStatus("sTrueVertexID", true);
    __flattree__->SetBranchStatus("sRun", true);
    __flattree__->SetBranchStatus("sSubRun", true);
    __tree_converter_reco_ttree__->SetBranchStatus("*", false);
    __tree_converter_reco_ttree__->SetBranchStatus("run", true);
    __tree_converter_reco_ttree__->SetBranchStatus("subrun", true);
    for(int i_entry = 0 ; i_entry < nb_entries ; i_entry++){ // genWeights
        GenericToolbox::display_loading(i_entry, nb_entries, Form("%sSynchronizing trees... (%i matches,%i miss)", WARNING.c_str(), nb_matches, nb_missing));
        __flattree__->GetEntry(i_entry);
        __tree_sync_cache__[i_entry] = -1;
        sTrueVertexID = __flattree__->GetLeaf("sTrueVertexID")->GetValue(0);
        int i_event = -1;
        for(auto &tree_converter_entry : __tree_converter_entries_mapping__){  // TreeConverter

            i_event = tree_converter_entry.first;

            if(__last_event_id_found__ > i_event) continue; // all events should be in order

            if( tree_converter_entry.second != sTrueVertexID )
                continue;
            else{

                __tree_converter_reco_ttree__->GetEntry(i_event); // load in memory

                if( // if it matches everything
                    __tree_converter_reco_ttree__->GetLeaf("run")->GetValue(0) == __flattree__->GetLeaf("sRun")->GetValue(0)
                    and __tree_converter_reco_ttree__->GetLeaf("subrun")->GetValue(0) == __flattree__->GetLeaf("sSubRun")->GetValue(0)
                    ){
                    __tree_sync_cache__[i_entry] = tree_converter_entry.first;
                    __last_event_id_found__ = tree_converter_entry.first;
                    break;
                }

            }
        }
        if(__tree_sync_cache__[i_entry] == -1){
            nb_missing++;
        }
        else {
            nb_matches++;
        }
    }
    __tree_converter_reco_ttree__->SetBranchStatus("*", true);
    __flattree__->SetBranchStatus("*", true);

    if(nb_matches == 0){
        std::cerr << ERROR << "Could not sync genWeights file and TreeConverter." << std::endl;
        exit(EXIT_FAILURE);
    }

    __tree_converter_entries_mapping__.clear();
    __tree_converter_entries_have_been_mapped__ = false;
    __tree_sync_cache_has_been_build__ = true;

}
void map_tree_converter_entries(){

    if(__tree_converter_entries_have_been_mapped__) return;

    std::cout << INFO << "Mapping Tree Converter File..." << std::endl;

    __tree_converter_reco_ttree__->SetBranchStatus("*", false);
    __tree_converter_reco_ttree__->SetBranchStatus("vertexID", true);
    for(int i_event = 0 ; i_event < __tree_converter_reco_ttree__->GetEntries(); i_event++){
        __tree_converter_reco_ttree__->GetEntry(i_event);
        __tree_converter_entries_mapping__[i_event] = Int_t(__tree_converter_reco_ttree__->GetLeaf("vertexID")->GetValue(0));
    }
    __tree_converter_reco_ttree__->SetBranchStatus("*", true);

    __tree_converter_entries_have_been_mapped__ = true;

}
void fill_component_mapping(){

    //Start with 2013.
    __from_Xsec_to_genWeights__["MAQE"] = "MAQE";
    __from_Xsec_to_genWeights__["MARES"] = "MARES";
    __from_Xsec_to_genWeights__["DISMPISHP"] = "DISMPISHP";
    __from_Xsec_to_genWeights__["SF"] = "SF";
    __from_Xsec_to_genWeights__["EB"] = "EB";
    __from_Xsec_to_genWeights__["PF"] = "PF";
    __from_Xsec_to_genWeights__["pF"] = "PF";
    __from_Xsec_to_genWeights__["PDD"] = "PDD";
    __from_Xsec_to_genWeights__["FSI_PI_ABS"] = "FSI_PI_ABS";
    __from_Xsec_to_genWeights__["FEFABS"] = "FSI_PI_ABS";
    __from_Xsec_to_genWeights__["FSI_PI_PROD"] = "FSI_PI_PROD";
    __from_Xsec_to_genWeights__["FEFINEL"] = "FSI_PI_PROD";
    __from_Xsec_to_genWeights__["FSI_INEL_LO"] = "FSI_INEL_LO";
    __from_Xsec_to_genWeights__["FEFQE"] = "FSI_INEL_LO";
    __from_Xsec_to_genWeights__["FSI_INEL_LO_E"] = "FSI_INEL_LO";
    __from_Xsec_to_genWeights__["FSI_INEL_HI"] = "FSI_INEL_HI";
    __from_Xsec_to_genWeights__["FEFQEH"] = "FSI_INEL_HI";
    __from_Xsec_to_genWeights__["FSI_INEL_HI_E"] = "FSI_INEL_HI";
    __from_Xsec_to_genWeights__["FSI_CEX_LO"] = "FSI_CEX_LO";
    __from_Xsec_to_genWeights__["FEFCX"] = "FSI_CEX_LO";
    __from_Xsec_to_genWeights__["FSI_CEX_LO_E"] = "FSI_CEX_LO";
    __from_Xsec_to_genWeights__["FSI_CEX_HI"] = "FSI_CEX_HI";
    __from_Xsec_to_genWeights__["FEFCXH"] = "FSI_CEX_HI";
    __from_Xsec_to_genWeights__["FSI_CEX_HI_E"] = "FSI_CEX_HI";
    __from_Xsec_to_genWeights__["PDD_MEC"] = "PDD_MEC";
    __from_Xsec_to_genWeights__["MEC"] = "MEC";

    //Now those added for 2014.
    __from_Xsec_to_genWeights__["MEC_C"] = "MEC_C";
    __from_Xsec_to_genWeights__["pF_C"] = "PF_C";
    __from_Xsec_to_genWeights__["EB_C"] = "EB_C";
    __from_Xsec_to_genWeights__["MEC_O"] = "MEC_O";
    __from_Xsec_to_genWeights__["pF_O"] = "PF_O";
    __from_Xsec_to_genWeights__["EB_O"] = "EB_O";
    __from_Xsec_to_genWeights__["CA5"] = "CA5";
    __from_Xsec_to_genWeights__["MANFFRES"] = "MARES";
    __from_Xsec_to_genWeights__["BgRES"] = "BgSclRes";
    __from_Xsec_to_genWeights__["SCCV"] = "SCCV";
    __from_Xsec_to_genWeights__["SCCA"] = "SCCA";
    __from_Xsec_to_genWeights__["RPA_C"] = "RPA";
    __from_Xsec_to_genWeights__["RPA_O"] = "RPA";  //TODO: This probably needs to be changed to separate these two... either that or handle them properly elsewhere.
    __from_Xsec_to_genWeights__["SF_RFG"] = "SF_RFG";
    __from_Xsec_to_genWeights__["CCNUE_0"] = "CCNuE";

    // Kendall's parameter
    __from_Xsec_to_genWeights__["COH_BS"] = "COH_BS";


    //Now those for 2016 with some additions and some names that have changed
    __from_Xsec_to_genWeights__["2p2h_shape_C"] = "MEC_shape_C";
    __from_Xsec_to_genWeights__["2p2h_shape_O"] = "MEC_shape_O";
    __from_Xsec_to_genWeights__["CC_DIS"] = "DISMPISHP";
    __from_Xsec_to_genWeights__["ISO_BKG"] = "BgSclRes";

    __from_Xsec_to_genWeights__["2p2h_Edep_lowEnu"] = "MEC_lowEnu";
    __from_Xsec_to_genWeights__["2p2h_Edep_highEnu"] = "MEC_highEnu";
    __from_Xsec_to_genWeights__["2p2h_Edep_lowEnubar"] = "MEC_lowEnubar";
    __from_Xsec_to_genWeights__["2p2h_Edep_highEnubar"] = "MEC_highEnubar";

    __from_Xsec_to_genWeights__["ISO_BKG_LowPPi"] = "BgSclRes_lowPPi";
    __from_Xsec_to_genWeights__["CC_BY_DIS"] = "DIS_BY_corr";
    __from_Xsec_to_genWeights__["CC_BY_MPi"] = "MultiPi_BY";
    __from_Xsec_to_genWeights__["CC_AGKY_Mult"] = "MultiPi_Xsec_AGKY";

    __from_Xsec_to_genWeights__["EB_bin_C_nu"] = "EB_bin_C_nu";
    __from_Xsec_to_genWeights__["EB_bin_O_nu"] = "EB_bin_O_nu";
    __from_Xsec_to_genWeights__["EB_bin_C_nubar"] = "EB_bin_C_nubar";
    __from_Xsec_to_genWeights__["EB_bin_O_nubar"] = "EB_bin_O_nubar";


    // https://github.com/t2k-software/T2KReWeight/blob/ecd103a17acfae04de001052b357fb07de364c58/app/genWeightsFromNRooTracker_BANFF_2020.cxx#L425-L465

    // CCQE:
//    __relative_variation_component_list__.emplace_back("kNXSec_MaCCQE");
    __relative_variation_component_list__.emplace_back("kNXSec_MaQE");

    // CC and NC single pion resonance:
    __relative_variation_component_list__.emplace_back("kNXSec_CA5RES");
    __relative_variation_component_list__.emplace_back("kNXSec_MaRES");

    // Use the separate iso half background dials
    __relative_variation_component_list__.emplace_back("kNXSec_BgSclRES");
    __relative_variation_component_list__.emplace_back("kNXSec_BgSclLMCPiBarRES");

    // All other CC and NC
    // Ed's CC DIS dials for 2020 Analysis
//    __relative_variation_component_list__.emplace_back("kNIWG_DIS_BY_corr");
//    __relative_variation_component_list__.emplace_back("kNIWG_MultiPi_BY_corr");
//    __relative_variation_component_list__.emplace_back("kNIWG_MultiPi_Xsec_AGKY");

    __relative_variation_component_list__.emplace_back("kNIWG_rpaCCQE_norm");
    __relative_variation_component_list__.emplace_back("kNIWG_rpaCCQE_shape");

    // FSI dials
    __relative_variation_component_list__.emplace_back("kNCasc_FrAbs_pi");
    __relative_variation_component_list__.emplace_back("kNCasc_FrCExLow_pi");
    __relative_variation_component_list__.emplace_back("kNCasc_FrInelLow_pi");
    __relative_variation_component_list__.emplace_back("kNCasc_FrPiProd_pi");
    __relative_variation_component_list__.emplace_back("kNCasc_FrInelHigh_pi");

    //-- PDD Weights, New Eb dial
    __relative_variation_component_list__.emplace_back("kNIWGMEC_PDDWeight_C12");
    __relative_variation_component_list__.emplace_back("kNIWGMEC_PDDWeight_O16");

    //2p2hEdep parameters
    __relative_variation_component_list__.emplace_back("kNIWG_2p2hEdep_lowEnu");
    __relative_variation_component_list__.emplace_back("kNIWG_2p2hEdep_highEnu");
    __relative_variation_component_list__.emplace_back("kNIWG_2p2hEdep_lowEnubar");
    __relative_variation_component_list__.emplace_back("kNIWG_2p2hEdep_highEnubar");


}
bool do_syst_param_has_relative_X_scale(std::string& syst_name_){
    bool is_relative_X = false;
    std::string genWeigths_name = __from_Xsec_to_genWeights__[syst_name_];
    if(genWeigths_name.empty()){
        genWeigths_name = syst_name_;
    }
    for(auto& relative_component_name: __relative_variation_component_list__){

        if(GenericToolbox::do_string_contains_substring(
            GenericToolbox::to_lower_case(relative_component_name),
            GenericToolbox::to_lower_case(genWeigths_name)
            )
           ){
//            std::cout << GenericToolbox::to_lower_case(relative_component_name) << " has ";
//            std::cout << GenericToolbox::to_lower_case(genWeigths_name) << " / " << syst_name_ << std::endl;
            is_relative_X = true;
            break;
        }
    }
//    std::cout << ALERT << syst_name_ << " is relative ? " << is_relative_X << std::endl;
    return is_relative_X;
}
void fill_parameter_nominal_values(){

//    __parameter_nominal_values_map__
    std::cout << WARNING << "Reading nominal value of parameters..." << std::endl;

    if(__covariance_matrix_file_path__.empty()){
        std::cerr << ERROR << "__covariance_matrix_file_path__ is not set." << std::endl;
        exit(EXIT_FAILURE);
    }

    TFile* covariance_matrix_tfile = TFile::Open(__covariance_matrix_file_path__.c_str(), "READ");
    if(not covariance_matrix_tfile->IsOpen()){
        std::cerr << ERROR << "Could not open : " << covariance_matrix_tfile->GetName() << std::endl;
        exit(EXIT_FAILURE);
    }

    __parameter_nominal_values_map__.clear();
    __parameter_error_map__.clear();

    auto* xsec_param_names = dynamic_cast<TObjArray *>(covariance_matrix_tfile->Get("xsec_param_names"));
    auto* xsec_param_nom_unnorm = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_nom_unnorm"));
    auto* xsec_cov_matrix = (TMatrixT<double> *)(covariance_matrix_tfile->Get("xsec_cov"));

    for(int i_parameter = 0 ; i_parameter < xsec_param_names->GetEntries() ; i_parameter++){
        __parameter_nominal_values_map__[xsec_param_names->At(i_parameter)->GetName()] = (*xsec_param_nom_unnorm)[i_parameter];
        __parameter_error_map__[xsec_param_names->At(i_parameter)->GetName()] = (*xsec_cov_matrix)[i_parameter][i_parameter];
    }



    covariance_matrix_tfile->Close();

}
double from_relative_par_value_to_absolute(std::string par_name_, double deviation_param_value_){

    double output;
    if(do_syst_param_has_relative_X_scale(par_name_)){
        // 0 -> nominal value
        // 0.1 -> +10% of the initial parameter
        output = __parameter_nominal_values_map__[par_name_]*(1 + deviation_param_value_);
    }
    else{
        // 0 -> nominal value
        // 0.1 -> +10% of 1 sigma
        output = __parameter_nominal_values_map__[par_name_] + deviation_param_value_*__parameter_error_map__[par_name_];
    }
    return output;
}
