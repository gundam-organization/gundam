//
// Created by Adrien BLANCHET on 16/09/2020.
//

// C++
#include "sstream"
#include "string"

// ROOT
#include "TFile.h"
#include "TTree.h"
#include "TMatrixT.h"
#include "TVectorT.h"
#include "TLeaf.h"
#include "TClonesArray.h"
#include "TTree.h"

// This project
#include "XsecDial.hh"
#include <FitStructs.hh>
#include <TGraph.h>
#include <TSpline.h>
#include <future>
#include "GlobalVariables.h"

// Submodules
#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolboxRootExt.h"

/****************************/
//! Globals
/****************************/
// User parameters
std::string __pathJsonConfigFile__;
std::string __pathCovarianceMatrixFile__;
std::string __pathTreeConverterFile__;
std::string __pathBinningFile__;
std::string __pathGenWeightsFileList__;
std::vector<std::string> __listGenWeightsFiles__;
bool __onlyRegenConfig__;
bool __skipReadSplines__;
bool __skipReadAndMergeSplines__;

// Internals
int __argc__;
int __lastTCEntryFound__ = 0;
char **__argv__;
std::string __commandLine__;
std::vector<xsllh::FitBin> __listKinematicBins__;
std::vector<std::string> __listRelativeVariationComponent__;
std::vector<std::string> __listSystematicNames__;
std::vector<std::string> __listSystematicNormSplinesNames__;
std::vector<std::string> __listSystematicXsecSplineNames__;
std::vector<std::string> __listSplitVarNames__;
std::map<std::string, double> __mapNominalValues__;
std::map<std::string, double> __mapErrorValues__;
std::map<std::string, std::string> __mapXsecToGenWeights__;
std::map<std::string, TFile*> __mapOutSplineTFiles__;
std::map<std::string, TTree*> __mapOutSplineTTrees__;
std::map<std::string, int> __toTopologyIndex__;
std::map<std::string, int> __toReactionIndex__;
std::map<int, int> __currentGenWeightToTreeConvEntry__;
std::map<int, Int_t> __treeConvEntryToVertexID__;
std::map<int, std::map<int, std::map<int, Int_t>>> __treeConvEntryToVertexIDSplit__; // run, subrun, entry, vertexID
TFile* __treeConverterTFile__;
TTree* __treeConverterRecoTTree__;
TFile* __currentGenWeightsTFile__;
TTree* __currentSampleSumTTree__;
TTree* __currentFlatTree__;

bool TEST__;

SplineBin __splineBinHandler__;

int __nbThreads__;

/****************************/
//! Subroutines
/****************************/
// Standard
std::string remindUsage();
void resetParameters();
void getUserParameters();
void initializeObjects();
void destroyObjects();

// Specific
void defineBinning(const std::string& binningFile_);
void fillComponentMapping();
void getListOfParameters();
void readInputCovarianceFile();
void fillDictionaries();

// Core
void binSplines();
void createNormSplines();
void readGenWeightsFiles();
void mergeSplines();
void processInterpolation();

bool buildTreeSyncCache();
void mapTreeConverterEntries();
void fillSplineBin(SplineBin& splineBin_, TTree* tree_);

// Config Generation
void generateJsonConfigFile();

double convertToAbsoluteVariation(std::string parName_, double relativeDeviationParamValue_);
bool doesParamHasRelativeXScale(std::string& systName_);


/****************************/
//! MAIN
/****************************/
int main(int argc, char** argv){

    LogDebug << "RAM: " << GenericToolbox::parseSizeUnits(GenericToolbox::getProcessMemoryUsage()) << std::endl;

    __argc__ = argc;
    __argv__ = argv;

    resetParameters();
    getUserParameters();
    remindUsage();

    // Setup
    initializeObjects();
    getListOfParameters();
    readInputCovarianceFile();
    fillDictionaries();

    // CORE
    if(not __onlyRegenConfig__) binSplines();

    generateJsonConfigFile();
    destroyObjects();

    LogDebug << "RAM: " << GenericToolbox::parseSizeUnits(GenericToolbox::getProcessMemoryUsage()) << std::endl;

    return EXIT_SUCCESS;
}


/****************************/
//! SUBROUTINES
/****************************/
std::string remindUsage(){

    std::stringstream remind_usage_ss;
    remind_usage_ss << "***********************************************" << std::endl;
    remind_usage_ss << " > Command Line Arguments" << std::endl;
    remind_usage_ss << "  -w : genweights input files (Current : " << GenericToolbox::parseVectorAsString(__listGenWeightsFiles__, true) << ")" << std::endl;
    remind_usage_ss << "  -l : genweights input file list (Current : " << __pathGenWeightsFileList__ << ")" << std::endl;
    remind_usage_ss << "  -b : binning file (Current : " << __pathBinningFile__ << ")" << std::endl;
    remind_usage_ss << "  -t : tree converter file (Current : " << __pathTreeConverterFile__ << ")" << std::endl;
    remind_usage_ss << "  -c : file containing infos on the covariance matrix (Current : " << __pathCovarianceMatrixFile__ << ")" << std::endl;
    remind_usage_ss << "  -mt : number of cores for reading files (Current : " << __nbThreads__ << ")" << std::endl;
    remind_usage_ss << "  --regen-config-only : Does not call binSplines, but only regen .json file (Current : " << __onlyRegenConfig__ << ")" << std::endl;
    remind_usage_ss << "  --skip-read-splines : Only merge and interpolate will be processed (Current : " << __skipReadSplines__ << ")" << std::endl;
    remind_usage_ss << "  --skip-read-merge-splines : Only interpolate will be processed (Current : " << __skipReadAndMergeSplines__ << ")" << std::endl;
    remind_usage_ss << "***********************************************" << std::endl;

    LogWarning << remind_usage_ss.str();
    return remind_usage_ss.str();

}
void resetParameters(){

    Logger::setUserHeaderStr("[xsllhGenWeightsFormater]");

    __pathJsonConfigFile__ = "";
    __pathCovarianceMatrixFile__ = "";
    __pathTreeConverterFile__ = "";
    __pathBinningFile__ = "";
    __pathGenWeightsFileList__ = "";

    __onlyRegenConfig__ = false;
    __skipReadSplines__ = false;
    __skipReadAndMergeSplines__ = false;

    __listSplitVarNames__ = {"beammode", "analysis", "fgd_reco", "cut_branch", "reaction"};

    __nbThreads__ = 1;

    __listGenWeightsFiles__.clear();
}
void getUserParameters(){

    if(__argc__ == 1){
        remindUsage();
        exit(EXIT_FAILURE);
    }

    LogWarning << "Sanity check" << std::endl;

    const std::string XSLLHFITTER = std::getenv("XSLLHFITTER");
    if(XSLLHFITTER.empty()){

        LogError << "Environment variable \"XSLLHFITTER\" not set." << std::endl
                 << "Cannot determine source tree location." << std::endl;
        remindUsage();
        exit(EXIT_FAILURE);
    }

    LogWarning << "Reading user parameters" << std::endl;

    __commandLine__ = GenericToolbox::joinVectorString(\
        std::vector<std::string>(__argv__ + 1, __argv__ + __argc__)
            , " ");

    for(int i_arg = 0; i_arg < __argc__; i_arg++){

        // Parameters
        if(std::string(__argv__[i_arg]) == "-j"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __pathJsonConfigFile__ = std::string(__argv__[j_arg]);
                if(not GenericToolbox::doesPathIsFile(__pathJsonConfigFile__)){
                    LogError << std::string(__argv__[i_arg]) << ": " << __pathJsonConfigFile__
                             << " could not be found." << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-w"){
            if (i_arg < __argc__ - 1) {
                do{
                    int j_arg = i_arg + 1;
                    if(not GenericToolbox::doesPathIsFile(std::string(__argv__[j_arg]))){
                        LogError << std::string(__argv__[i_arg]) << ": " << std::string(__argv__[j_arg]) << " could not be found." << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    __listGenWeightsFiles__.emplace_back(std::string(__argv__[j_arg]));
                    if(__argv__[j_arg+1][0] == '-') break;
                    i_arg++; // changing i_arg for next loop
                } while(i_arg < __argc__ - 1);
            } else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-b"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __pathBinningFile__ = std::string(__argv__[j_arg]);
                if(not GenericToolbox::doesPathIsFile(__pathBinningFile__)){
                    LogError << std::string(__argv__[i_arg]) << ": " << __pathBinningFile__
                             << " could not be found." << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-t"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __pathTreeConverterFile__ = std::string(__argv__[j_arg]);
                if(not GenericToolbox::doesPathIsFile(__pathTreeConverterFile__)){
                    LogError << std::string(__argv__[i_arg]) << ": " << __pathCovarianceMatrixFile__
                             << " could not be found." << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-c"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __pathCovarianceMatrixFile__ = std::string(__argv__[j_arg]);
                if(not GenericToolbox::doesPathIsFile(__pathCovarianceMatrixFile__)){
                    LogError << std::string(__argv__[i_arg]) << ": " << __pathCovarianceMatrixFile__
                             << " could not be found." << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-l"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __pathGenWeightsFileList__ = std::string(__argv__[j_arg]);
                if(not GenericToolbox::doesPathIsFile(__pathGenWeightsFileList__)){
                    LogError << std::string(__argv__[i_arg]) << ": " << __pathGenWeightsFileList__
                             << " could not be found." << std::endl;
                    exit(EXIT_FAILURE);
                }
                auto fileList = GenericToolbox::dumpFileAsVectorString(__pathGenWeightsFileList__);
                __listGenWeightsFiles__.insert(
                    __listGenWeightsFiles__.end(),
                    fileList.begin(),
                    fileList.end()
                    );
            } else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-mt"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __nbThreads__ = std::stoi(__argv__[j_arg]);
            } else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }

        // Triggers
        else if(std::string(__argv__[i_arg]) == "--regen-config-only"){
            __onlyRegenConfig__ = true;
        }
        else if(std::string(__argv__[i_arg]) == "--skip-read-splines"){
            __skipReadSplines__ = true;
        }
        else if(std::string(__argv__[i_arg]) == "--skip-read-merge-splines"){
            __skipReadAndMergeSplines__ = true;
        }

    }

}
void initializeObjects(){

    LogInfo << "Building component name mapping from Xsec to GenWeights conventions..." << std::endl;
    fillComponentMapping();

    LogInfo << "Defining the binning..." << std::endl;
    if(__pathBinningFile__.empty()){
        LogError << "__binning_file_path__ not set." << std::endl;
        exit(EXIT_FAILURE);
    }
    defineBinning(__pathBinningFile__);

    LogInfo << "Opening TreeConverter file..." << std::endl;
    if(__pathTreeConverterFile__.empty()){
        LogError << "__tree_converter_file_path__ not set." << std::endl;
        exit(EXIT_FAILURE);
    }
    __treeConverterTFile__        = TFile::Open(__pathTreeConverterFile__.c_str(), "READ");
    __treeConverterRecoTTree__    = (TTree*) __treeConverterTFile__->Get("selectedEvents");

    LogInfo << "Claiming memory slots for the bin handler..." << std::endl;
    for(const auto& splitVarName: __listSplitVarNames__){
        __splineBinHandler__.splitVarNameList.emplace_back(splitVarName);
        __splineBinHandler__.splitVarValueList.emplace_back(-1);
    }
    __splineBinHandler__.reset();

}
void destroyObjects(){

//    if(__input_genWeights_tfile__ != nullptr) __input_genWeights_tfile__->Close();
    if(__treeConverterTFile__ != nullptr) __treeConverterTFile__->Close();

}


void defineBinning(const std::string& binningFile_) {

    std::ifstream fin(binningFile_, std::ios::in);
    if(!fin.is_open())
    {
        LogError << "Can't open binning file." << std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        std::string line;
        while(std::getline(fin, line)) {
            std::stringstream ss(line);
            double D1_1, D1_2, D2_1, D2_2;
            if(!(ss >> D2_1 >> D2_2 >> D1_1 >> D1_2))
            {
                LogWarning << "Bad line format: " << line << std::endl;
                continue;
            }
            __listKinematicBins__.emplace_back(xsllh::FitBin(D1_1, D1_2, D2_1, D2_2));
        }
    }
    fin.close();

}
void fillComponentMapping() {

    //Start with 2013.
    __mapXsecToGenWeights__["MAQE"] = "MAQE";
    __mapXsecToGenWeights__["MARES"] = "MARES";
    __mapXsecToGenWeights__["DISMPISHP"] = "DISMPISHP";
    __mapXsecToGenWeights__["SF"] = "SF";
    __mapXsecToGenWeights__["EB"] = "EB";
    __mapXsecToGenWeights__["PF"] = "PF";
    __mapXsecToGenWeights__["pF"] = "PF";
    __mapXsecToGenWeights__["PDD"] = "PDD";
    __mapXsecToGenWeights__["FSI_PI_ABS"] = "FSI_PI_ABS";
    __mapXsecToGenWeights__["FEFABS"] = "FSI_PI_ABS";
    __mapXsecToGenWeights__["FSI_PI_PROD"] = "FSI_PI_PROD";
    __mapXsecToGenWeights__["FEFINEL"] = "FSI_PI_PROD";
    __mapXsecToGenWeights__["FSI_INEL_LO"] = "FSI_INEL_LO";
    __mapXsecToGenWeights__["FEFQE"] = "FSI_INEL_LO";
    __mapXsecToGenWeights__["FSI_INEL_LO_E"] = "FSI_INEL_LO";
    __mapXsecToGenWeights__["FSI_INEL_HI"] = "FSI_INEL_HI";
    __mapXsecToGenWeights__["FEFQEH"] = "FSI_INEL_HI";
    __mapXsecToGenWeights__["FSI_INEL_HI_E"] = "FSI_INEL_HI";
    __mapXsecToGenWeights__["FSI_CEX_LO"] = "FSI_CEX_LO";
    __mapXsecToGenWeights__["FEFCX"] = "FSI_CEX_LO";
    __mapXsecToGenWeights__["FSI_CEX_LO_E"] = "FSI_CEX_LO";
    __mapXsecToGenWeights__["FSI_CEX_HI"] = "FSI_CEX_HI";
    __mapXsecToGenWeights__["FEFCXH"] = "FSI_CEX_HI";
    __mapXsecToGenWeights__["FSI_CEX_HI_E"] = "FSI_CEX_HI";
    __mapXsecToGenWeights__["PDD_MEC"] = "PDD_MEC";
    __mapXsecToGenWeights__["MEC"] = "MEC";

    //Now those added for 2014.
    __mapXsecToGenWeights__["MEC_C"] = "MEC_C";
    __mapXsecToGenWeights__["pF_C"] = "PF_C";
    __mapXsecToGenWeights__["EB_C"] = "EB_C";
    __mapXsecToGenWeights__["MEC_O"] = "MEC_O";
    __mapXsecToGenWeights__["pF_O"] = "PF_O";
    __mapXsecToGenWeights__["EB_O"] = "EB_O";
    __mapXsecToGenWeights__["CA5"] = "CA5";
    __mapXsecToGenWeights__["MANFFRES"] = "MARES";
    __mapXsecToGenWeights__["BgRES"] = "BgSclRes";
    __mapXsecToGenWeights__["SCCV"] = "SCCV";
    __mapXsecToGenWeights__["SCCA"] = "SCCA";
    __mapXsecToGenWeights__["RPA_C"] = "RPA";
    __mapXsecToGenWeights__["RPA_O"] = "RPA";  //TODO: This probably needs to be changed to separate these two... either that or handle them properly elsewhere.
    __mapXsecToGenWeights__["SF_RFG"] = "SF_RFG";
    __mapXsecToGenWeights__["CCNUE_0"] = "CCNuE";

    // Kendall's parameter
    __mapXsecToGenWeights__["COH_BS"] = "COH_BS";


    //Now those for 2016 with some additions and some names that have changed
    __mapXsecToGenWeights__["2p2h_shape_C"] = "MEC_shape_C";
    __mapXsecToGenWeights__["2p2h_shape_O"] = "MEC_shape_O";
    __mapXsecToGenWeights__["CC_DIS"] = "DISMPISHP";
    __mapXsecToGenWeights__["ISO_BKG"] = "BgSclRes";

    __mapXsecToGenWeights__["2p2h_Edep_lowEnu"] = "MEC_lowEnu";
    __mapXsecToGenWeights__["2p2h_Edep_highEnu"] = "MEC_highEnu";
    __mapXsecToGenWeights__["2p2h_Edep_lowEnubar"] = "MEC_lowEnubar";
    __mapXsecToGenWeights__["2p2h_Edep_highEnubar"] = "MEC_highEnubar";

    __mapXsecToGenWeights__["ISO_BKG_LowPPi"] = "BgSclRes_lowPPi";
    __mapXsecToGenWeights__["CC_BY_DIS"] = "DIS_BY_corr";
    __mapXsecToGenWeights__["CC_BY_MPi"] = "MultiPi_BY";
    __mapXsecToGenWeights__["CC_AGKY_Mult"] = "MultiPi_Xsec_AGKY";

    __mapXsecToGenWeights__["EB_bin_C_nu"] = "EB_bin_C_nu";
    __mapXsecToGenWeights__["EB_bin_O_nu"] = "EB_bin_O_nu";
    __mapXsecToGenWeights__["EB_bin_C_nubar"] = "EB_bin_C_nubar";
    __mapXsecToGenWeights__["EB_bin_O_nubar"] = "EB_bin_O_nubar";


    // https://github.com/t2k-software/T2KReWeight/blob/ecd103a17acfae04de001052b357fb07de364c58/app/genWeightsFromNRooTracker_BANFF_2020.cxx#L425-L465

    // CCQE:
//    __relative_variation_component_list__.emplace_back("kNXSec_MaCCQE");
    __listRelativeVariationComponent__.emplace_back("kNXSec_MaQE");

    // CC and NC single pion resonance:
    __listRelativeVariationComponent__.emplace_back("kNXSec_CA5RES");
    __listRelativeVariationComponent__.emplace_back("kNXSec_MaRES");

    // Use the separate iso half background dials
    __listRelativeVariationComponent__.emplace_back("kNXSec_BgSclRES");
    __listRelativeVariationComponent__.emplace_back("kNXSec_BgSclLMCPiBarRES");

    // All other CC and NC
    // Ed's CC DIS dials for 2020 Analysis
//    __relative_variation_component_list__.emplace_back("kNIWG_DIS_BY_corr");
//    __relative_variation_component_list__.emplace_back("kNIWG_MultiPi_BY_corr");
//    __relative_variation_component_list__.emplace_back("kNIWG_MultiPi_Xsec_AGKY");

    __listRelativeVariationComponent__.emplace_back("kNIWG_rpaCCQE_norm");
    __listRelativeVariationComponent__.emplace_back("kNIWG_rpaCCQE_shape");

    // FSI dials
    __listRelativeVariationComponent__.emplace_back("kNCasc_FrAbs_pi");
    __listRelativeVariationComponent__.emplace_back("kNCasc_FrCExLow_pi");
    __listRelativeVariationComponent__.emplace_back("kNCasc_FrInelLow_pi");
    __listRelativeVariationComponent__.emplace_back("kNCasc_FrPiProd_pi");
    __listRelativeVariationComponent__.emplace_back("kNCasc_FrInelHigh_pi");

    //-- PDD Weights, New Eb dial
    __listRelativeVariationComponent__.emplace_back("kNIWGMEC_PDDWeight_C12");
    __listRelativeVariationComponent__.emplace_back("kNIWGMEC_PDDWeight_O16");

    //2p2hEdep parameters
    __listRelativeVariationComponent__.emplace_back("kNIWG_2p2hEdep_lowEnu");
    __listRelativeVariationComponent__.emplace_back("kNIWG_2p2hEdep_highEnu");
    __listRelativeVariationComponent__.emplace_back("kNIWG_2p2hEdep_lowEnubar");
    __listRelativeVariationComponent__.emplace_back("kNIWG_2p2hEdep_highEnubar");

}
void getListOfParameters() {

    LogWarning << "Getting list of parameters..." << std::endl;

    if(__pathCovarianceMatrixFile__.empty()){
        LogError << "__covariance_matrix_file_path__ is not set." << std::endl;
        exit(EXIT_FAILURE);
    }

    TFile* covariance_matrix_tfile = TFile::Open(__pathCovarianceMatrixFile__.c_str(), "READ");
    if(not covariance_matrix_tfile->IsOpen()){
        LogError << "Could not open : " << covariance_matrix_tfile->GetName() << std::endl;
        exit(EXIT_FAILURE);
    }

    auto* xsec_param_names = dynamic_cast<TObjArray *>(covariance_matrix_tfile->Get("xsec_param_names"));
    for(int i_parameter = 0 ; i_parameter < xsec_param_names->GetEntries() ; i_parameter++){

        __listSystematicNames__.emplace_back(xsec_param_names->At(i_parameter)->GetName());

        if(not __mapXsecToGenWeights__[xsec_param_names->At(i_parameter)->GetName()].empty()){
            LogInfo << "  -> GenWeights component : \"" << __mapXsecToGenWeights__[xsec_param_names->At(i_parameter)->GetName()];
            LogInfo << "\" has been identified as " << xsec_param_names->At(i_parameter)->GetName() << "." << std::endl;
        }
        else {
            LogAlert << "  -> No equivalent for " << xsec_param_names->At(i_parameter)->GetName() << " in GenWeights. Will be treated as a norm factor." << std::endl;
            __listSystematicNormSplinesNames__.emplace_back(xsec_param_names->At(i_parameter)->GetName());
//            __missing_splines__.emplace_back(xsec_param_names->At(i_parameter)->GetName());
            continue;
        }

        __listSystematicXsecSplineNames__.emplace_back(xsec_param_names->At(i_parameter)->GetName());

    }

    covariance_matrix_tfile->Close();

    LogInfo << "List of systematics to process:" << std::endl;
    for( auto & systName : __listSystematicNames__){
        if(doesParamHasRelativeXScale(systName)) LogAlert << "  - " << systName << " (Relative)" << std::endl;
        else LogInfo << "  - " << systName << std::endl;
    }

}
void readInputCovarianceFile() {

    LogWarning << "Reading nominal value of parameters..." << std::endl;

    if(__pathCovarianceMatrixFile__.empty()){
        LogError << "__covariance_matrix_file_path__ is not set." << std::endl;
        exit(EXIT_FAILURE);
    }

    TFile* covariance_matrix_tfile = TFile::Open(__pathCovarianceMatrixFile__.c_str(), "READ");
    if(not covariance_matrix_tfile->IsOpen()){
        LogError << "Could not open : " << covariance_matrix_tfile->GetName() << std::endl;
        exit(EXIT_FAILURE);
    }

    __mapNominalValues__.clear();
    __mapErrorValues__.clear();

    auto* xsec_param_names = dynamic_cast<TObjArray *>(covariance_matrix_tfile->Get("xsec_param_names"));
    auto* xsec_param_nom_unnorm = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_nom_unnorm"));
    auto* xsec_cov_matrix = (TMatrixT<double> *)(covariance_matrix_tfile->Get("xsec_cov"));

    for(int i_parameter = 0 ; i_parameter < xsec_param_names->GetEntries() ; i_parameter++){
        __mapNominalValues__[xsec_param_names->At(i_parameter)->GetName()] = (*xsec_param_nom_unnorm)[i_parameter];
        __mapErrorValues__[xsec_param_names->At(i_parameter)->GetName()] = (*xsec_cov_matrix)[i_parameter][i_parameter];
    }

    covariance_matrix_tfile->Close();

}
void fillDictionaries(){

    __toTopologyIndex__["CC-0pi"] = 0;
    __toTopologyIndex__["CC-1pi"] = 1;
    __toTopologyIndex__["CC-Other"] = 2;

    __toReactionIndex__["CCQE"] = 0;
    __toReactionIndex__["2p2h"] = 9;
    __toReactionIndex__["RES"] = 1;
    __toReactionIndex__["DIS"] = 2;
    __toReactionIndex__["Coh"] = 3;
    __toReactionIndex__["NC"] = 4;
    __toReactionIndex__["CC-#bar{#nu}_{#mu}"] = 5;
    __toReactionIndex__["CC-#nu_{e}, CC-#bar{#nu}_{e}"] = 6;
    __toReactionIndex__["out FV"] = 7;
    __toReactionIndex__["other"] = 999;
    __toReactionIndex__["no truth"] = -1;
    __toReactionIndex__["sand #mu"] = 777;

}


void binSplines(){

    // Read splines
    if(not __skipReadSplines__ and not __skipReadAndMergeSplines__) readGenWeightsFiles();

    // Merge splines
    if(not __skipReadAndMergeSplines__) mergeSplines();

    // Process interpolation between kinematic bins
    processInterpolation();

}
void readGenWeightsFiles(){

    LogWarning << "Reading GenWeights Files..." << std::endl;

    LogInfo << "Creating Output Splines Files..." << std::endl;
    for(const auto& xsecSplineName : __listSystematicXsecSplineNames__){
        __mapOutSplineTFiles__[xsecSplineName] = TFile::Open(
            Form("%s_splines.root", xsecSplineName.c_str()),
            "RECREATE"
        );
        __mapOutSplineTTrees__[xsecSplineName + "_Unbinned"] = new TTree("UnbinnedSplines", "UnbinnedSplines");
    }

    LogInfo << "Hooking Output Unbinned Splines TTree" << std::endl;
    for( auto & systName : __listSystematicXsecSplineNames__){
        for( int iSplitVar = 0 ; iSplitVar < int(__listSplitVarNames__.size()) ; iSplitVar++ ){
            __mapOutSplineTTrees__[systName + "_Unbinned"]->Branch(
                __splineBinHandler__.splitVarNameList[iSplitVar].c_str(), &__splineBinHandler__.splitVarValueList[iSplitVar]
            );
        }
        __mapOutSplineTTrees__[systName + "_Unbinned"]->Branch("D1Reco", &__splineBinHandler__.D1Reco);
        __mapOutSplineTTrees__[systName + "_Unbinned"]->Branch("D2Reco", &__splineBinHandler__.D2Reco);
        __mapOutSplineTTrees__[systName + "_Unbinned"]->Branch("kinematicBin", &__splineBinHandler__.kinematicBin);
        __mapOutSplineTTrees__[systName + "_Unbinned"]->Branch("graph", &__splineBinHandler__.graphHandler);
        __mapOutSplineTTrees__[systName + "_Unbinned"]->Branch("spline", &__splineBinHandler__.splinePtr);
    }

    LogInfo << "Reading genWeights input files..." << std::endl;
    int nbGenWightsFiles = __listGenWeightsFiles__.size();

    std::function<void(int)> readInputFilesFunction = [nbGenWightsFiles](int iThread_){

        bool isMultiThreaded = (iThread_ != -1);
        TFile* genWeightsTFilePtr = nullptr;
        TTree* sampleSumTTreePtr = nullptr;
        TTree* flatTTreePtr = nullptr;
        TTree* treeConverterRecoTTree    = (TTree*) __treeConverterTFile__->Get("selectedEvents");
        SplineBin splineBinHandler;

        for(int iFile = 0 ; iFile < nbGenWightsFiles; iFile++){

            if(isMultiThreaded and iFile% __nbThreads__ != iThread_){
                continue;
            }

            GlobalVariables::getThreadMutex().lock();
            GenericToolbox::displayProgressBar(iFile, nbGenWightsFiles, LogWarning.getPrefixString()+"Reading genWeights files...");
            LogAlert << "Opening genWeights file " << GenericToolbox::splitString(__listGenWeightsFiles__.at(iFile), "/").back();
            LogAlert << " " << iFile+1 << "/" << int(__listGenWeightsFiles__.size()) << "..." << std::endl;

            if( not GenericToolbox::doesTFileIsValid(__listGenWeightsFiles__.at(iFile)) ){
                LogError << __listGenWeightsFiles__.at(iFile) << " can't be opened. Skipping..." << std::endl;
                continue; // skip
            }
            GlobalVariables::getThreadMutex().unlock();

            // Closing the previously opened genWeights file
            if(genWeightsTFilePtr != nullptr){
                genWeightsTFilePtr->Close();
                delete genWeightsTFilePtr;
                sampleSumTTreePtr = nullptr;
                flatTTreePtr = nullptr;
            }
            genWeightsTFilePtr = TFile::Open(__listGenWeightsFiles__.at(iFile).c_str(), "READ");
            sampleSumTTreePtr  = (TTree*) genWeightsTFilePtr->Get("sample_sum");
            flatTTreePtr       = (TTree*) genWeightsTFilePtr->Get("flattree"); // BUGGY LINE IN MULTI THREAD... -> strange segfault


            if(sampleSumTTreePtr == nullptr or flatTTreePtr == nullptr){
                LogError << "Could not find needed TTrees: ";
                LogError << "flatTTreePtr=" << flatTTreePtr;
                LogError << "sampleSumTTreePtr=" << sampleSumTTreePtr << std::endl;
                LogError << "Skipping..." << std::endl;
                continue;
            }


            GlobalVariables::getThreadMutex().lock();
            __currentSampleSumTTree__ = sampleSumTTreePtr;
            __currentFlatTree__ = flatTTreePtr;
            if(not buildTreeSyncCache()){
                GlobalVariables::getThreadMutex().unlock();
                continue;
            }
            auto currentGenWeightToTreeConvEntry = __currentGenWeightToTreeConvEntry__; // copy
            GlobalVariables::getThreadMutex().unlock();

            LogInfo << "Grabbing Splines Graphs..." << std::endl;
            sampleSumTTreePtr->GetEntry(0);
            int nbOfSamples = sampleSumTTreePtr->GetLeaf("NSamples")->GetValue(0);
            std::map<std::string, TClonesArray*> clone_array_map;
            GenericToolbox::muteRoot();
            for( auto &syst_name : __listSystematicXsecSplineNames__ ){

                clone_array_map[syst_name] = new TClonesArray("TGraph", nbOfSamples);

                sampleSumTTreePtr->SetBranchAddress(
                    Form("%sGraph", __mapXsecToGenWeights__[syst_name].c_str()),
                    &clone_array_map[syst_name]
                );

            }
            GenericToolbox::unmuteRoot();



            LogInfo << "Looping over all the genWeights entries..." << std::endl;
            int nbGenWeightsEntries = sampleSumTTreePtr->GetEntries();
            int nbGraphsFound = 0;
            for(int iGenWeightsEntry = 0 ; iGenWeightsEntry < nbGenWeightsEntries; iGenWeightsEntry++){

                if(currentGenWeightToTreeConvEntry[iGenWeightsEntry] == -1){
                    continue;
                }

                sampleSumTTreePtr->GetEntry(iGenWeightsEntry);
                treeConverterRecoTTree->GetEntry(currentGenWeightToTreeConvEntry[iGenWeightsEntry]);

                GlobalVariables::getThreadMutex().lock();
                fillSplineBin(__splineBinHandler__, treeConverterRecoTTree);

                if(__splineBinHandler__.kinematicBin == -1){
                    GlobalVariables::getThreadMutex().unlock();
                    continue;
                }

                for( auto & systName : __listSystematicXsecSplineNames__){

                    for(int iSample = 0 ; iSample < nbOfSamples ; iSample++ ){

                        if(iSample >= clone_array_map[systName]->GetSize() ) continue;

                        auto* newGraph = (TGraph*)(clone_array_map[systName]->At(iSample));

                        if( clone_array_map[systName]->At(iSample) != nullptr
                            and newGraph->GetN() > 1
                            ){

                            bool splines_are_non_zeroed = false;
                            for(int i_point = 0 ; i_point < newGraph->GetN() ; i_point++){
                                if(newGraph->GetY()[i_point] != 0){
                                    splines_are_non_zeroed = true;
                                    break;
                                }
                            }
                            if(not splines_are_non_zeroed)
                                continue;

                            std::string binName = "graph_";
                            binName += __splineBinHandler__.generateBinName();

                            newGraph->SetName(binName.c_str());
                            newGraph->SetTitle(binName.c_str());
                            newGraph->SetMarkerStyle(kFullDotLarge);
                            newGraph->SetMarkerSize(1);
                            nbGraphsFound++;

                            __splineBinHandler__.graphHandler = (TGraph*)newGraph->Clone();
                            __splineBinHandler__.splinePtr
                                = new TSpline3(
                                __splineBinHandler__.graphHandler->GetName(), __splineBinHandler__.graphHandler );

                            __mapOutSplineTTrees__[systName + "_Unbinned"]->Fill();

                        }
                    }

                }
                GlobalVariables::getThreadMutex().unlock();

            } // i_entry

            LogInfo << "Freeing up memory..." << std::endl;
            LogDebug << "RAM used before: " << GenericToolbox::parseSizeUnits(GenericToolbox::getProcessMemoryUsage()) << std::endl;
            for( auto &syst_name : __listSystematicXsecSplineNames__){
                delete clone_array_map[syst_name];
            }
            LogDebug << "RAM used after: " << GenericToolbox::parseSizeUnits(GenericToolbox::getProcessMemoryUsage()) << std::endl;

            LogInfo.quietLineJump();

        }
    };

    if(__nbThreads__ > 1){
        std::vector<std::future<void>> threadsList;
        for( int iThread = 0 ; iThread < __nbThreads__; iThread++ ){
            threadsList.emplace_back(
                std::async( std::launch::async, std::bind(readInputFilesFunction, iThread) )
            );
        }

        std::string progressBarPrefix = LogWarning.getPrefixString() + "Reading events...";
        for( int iThread = 0 ; iThread < __nbThreads__; iThread++ ){
            while(threadsList[iThread].wait_for(std::chrono::milliseconds(33)) != std::future_status::ready){
//                GenericToolbox::displayProgressBar(*counterPtr, totalNbEventsToLoad, progressBarPrefix);
            }
            threadsList[iThread].get();
        }
    }
    else{
        readInputFilesFunction(-1);
    }

    LogInfo << "Writing Unbinned Splines TTrees..." << std::endl;
    for( auto &syst_name : __listSystematicXsecSplineNames__){

        __mapOutSplineTFiles__[syst_name]->WriteObject(__mapOutSplineTTrees__[syst_name+"_Unbinned"], "UnbinnedSplines");
        __mapOutSplineTFiles__[syst_name]->Close();
        delete __mapOutSplineTFiles__[syst_name];
        __mapOutSplineTFiles__[syst_name] = nullptr;

    }

}
void mergeSplines(){

    LogWarning << "Averaging splines in bins..." << std::endl;

    // Holders
    std::vector<SplineBin> processedBins;
    std::vector<std::vector<TGraph*>> graphsListHolder;
    TTree* unbinnedSplinesTreeBuf = nullptr;
    SplineBin currentEntryBin;
    for(int iSplitVar = 0 ; iSplitVar < int(__listSplitVarNames__.size()) ; iSplitVar++){
        currentEntryBin.addSplitVar(__listSplitVarNames__[iSplitVar], -1);
    }

    GenericToolbox::getProcessMemoryUsageDiffSinceLastCall(); // init
    int nbXsecSplines = __listSystematicXsecSplineNames__.size();
    for( int iXsec = 0 ; iXsec < nbXsecSplines ; iXsec++ ){

        std::string xsecSplineName = __listSystematicXsecSplineNames__[iXsec];

        // Holders
        processedBins.clear();
        graphsListHolder.clear();
        currentEntryBin.reset();

        __mapOutSplineTFiles__[xsecSplineName] = TFile::Open(
            Form("%s_splines.root", xsecSplineName.c_str()),
            "READ"
        );
        unbinnedSplinesTreeBuf = (TTree*) __mapOutSplineTFiles__[xsecSplineName]->Get("UnbinnedSplines");
        int nbEntries = unbinnedSplinesTreeBuf->GetEntries();

        LogInfo << "Gathering " << xsecSplineName << " splines. (" << nbEntries << " graphs)" << std::endl;

        // Hooking the spline bin the TTree
        unbinnedSplinesTreeBuf->SetBranchAddress("kinematicBin", &currentEntryBin.kinematicBin);
        for(int iSplitVar = 0 ; iSplitVar < int(__listSplitVarNames__.size()) ; iSplitVar++){
            unbinnedSplinesTreeBuf->SetBranchAddress(
                currentEntryBin.splitVarNameList[iSplitVar].c_str(),
                &currentEntryBin.splitVarValueList[iSplitVar]
                );
        }
        unbinnedSplinesTreeBuf->SetBranchAddress("graph", &currentEntryBin.graphHandler);

        // Reading the tree
        std::string progressTitle = LogWarning.getPrefixString() + "Gathering " + xsecSplineName + " splines";
        for( int iSplineEntry = 0 ; iSplineEntry < nbEntries ; iSplineEntry++ ){

            GenericToolbox::displayProgressBar(iSplineEntry, nbEntries, progressTitle);

            unbinnedSplinesTreeBuf->GetEntry(iSplineEntry);

            // Fast skip
            if(   currentEntryBin.kinematicBin <  0
               or currentEntryBin.kinematicBin >= __listKinematicBins__.size()) continue;

            int index = -1;
            std::string currentBinName = currentEntryBin.generateBinName();
            for(size_t iProcessedBin = 0 ; iProcessedBin < processedBins.size() ; iProcessedBin++){
                if(processedBins[iProcessedBin].generateBinName() == currentBinName){
                    index = iProcessedBin;
                    break;
                }
            }

            if(index == -1){ // NEW BIN
                processedBins.emplace_back(currentEntryBin); // copied
                graphsListHolder.emplace_back(std::vector<TGraph*>());
                index = graphsListHolder.size()-1;
            }

            graphsListHolder[index].emplace_back((TGraph*) currentEntryBin.graphHandler->Clone()); // add the graph to the current bin
            graphsListHolder[index].back()->SetTitle(
                Form("%s_%i", currentBinName.c_str()
                     , int(graphsListHolder[index].size())));
            graphsListHolder[index].back()->SetName(
                Form("%s_%i", currentBinName.c_str()
                    , int(graphsListHolder[index].size())));

            // CHECK IF FLAT
//            double Ycheck = -1;
//            for(int i_pt = 0 ; i_pt < currentEntryBin.graphHandler->GetN() ; i_pt++){
//
//                if(Ycheck == -1){
//                    Ycheck = currentEntryBin.graphHandler->GetY()[i_pt];
//                }
//                else if(Ycheck != currentEntryBin.graphHandler->GetY()[i_pt]){
//                    break;
//                }
//                else if(i_pt+1 == currentEntryBin.graphHandler->GetN()){
//                    LogAlert << xsecSplineName << ": " << currentEntryBin.generateBinName() << " is FLAT." << std::endl;
//                }
//
//            }


        } // iSplineEntry

        if(processedBins.empty()){
            LogError << "No spline found for " << xsecSplineName << std::endl;
            continue;
        }

        // Merging splines
        LogInfo << "Now merging splines (" << processedBins.size() << " bins to process)" << std::endl;
        for( int iBin = 0 ; iBin < int(processedBins.size()) ; iBin++ ){

            std::vector<double> X;
            std::vector<double> Y;

            // look for all sampled X points
            for( const auto &graph : graphsListHolder[iBin] ){
                for( int i_pt = 0 ; i_pt < graph->GetN() ; i_pt++ ){
                    if(not GenericToolbox::doesElementIsInVector( graph->GetX()[i_pt], X )){
                        X.emplace_back(graph->GetX()[i_pt]);
                    }
                }
            }

            // Averaging Y points
            for( int iX = 0 ; iX < int(X.size())  ; iX++ ){

                Y.emplace_back(0);
                int nbSamples = 0; // count how many time iX has been measured
                for(auto &graph : graphsListHolder[iBin]){
                    for(int i_pt = 0 ; i_pt < graph->GetN() ; i_pt++){
                        if(graph->GetX()[i_pt] == X[iX]){
                            Y.back() += graph->GetY()[i_pt];
                            nbSamples++;
                        }
                    }
                }
                Y.back() /= double(nbSamples);

                // convert to absolute syst parameter value
                X[iX] = convertToAbsoluteVariation(xsecSplineName, X[iX]);

            }

            processedBins[iBin].graphHandler  = new TGraph( X.size(), &X[0], &Y[0] );
            processedBins[iBin].splinePtr
                = new TSpline3(processedBins[iBin].generateBinName().c_str(), processedBins[iBin].graphHandler );

        } // iBin

        // Sort
        LogInfo << "Sorting bins by name" << std::endl;
        std::function<bool(SplineBin, SplineBin)> aGoesFirst = [](const SplineBin& a, const SplineBin& b){
            for(int iSplitVar = 0 ; iSplitVar < int(__listSplitVarNames__.size()) ; iSplitVar++){
                if(a.splitVarValueList[iSplitVar] != b.splitVarValueList[iSplitVar]){
                    return a.splitVarValueList[iSplitVar] < b.splitVarValueList[iSplitVar];
                }
            }
            if( a.kinematicBin != b.kinematicBin ) return a.kinematicBin < b.kinematicBin;
            return false; // arbitrary
        };
//        std::sort(processedBins.begin(), processedBins.end(), aGoesFirst);
        auto p = GenericToolbox::getSortPermutation(processedBins, aGoesFirst);
        processedBins    = GenericToolbox::applyPermutation(processedBins, p);
        graphsListHolder = GenericToolbox::applyPermutation(graphsListHolder, p);

        // Reopening
        __mapOutSplineTFiles__[xsecSplineName]->Close();
        delete __mapOutSplineTFiles__[xsecSplineName];
        __mapOutSplineTFiles__[xsecSplineName] = TFile::Open(
            Form("%s_splines.root", xsecSplineName.c_str()),
            "UPDATE"
        );
        __mapOutSplineTTrees__[xsecSplineName + "_Binned"] = new TTree("BinnedSplines", "BinnedSplines");

        TDirectory* unbinnedSplinesDirectory = GenericToolbox::mkdirTFile(__mapOutSplineTFiles__[xsecSplineName], "UnbinnedSplinesPlots");

        for( int iBin = 0 ; iBin < int(processedBins.size()) ; iBin++ ){
            unbinnedSplinesDirectory->cd();
            std::string binName = processedBins[iBin].generateBinName();
            GenericToolbox::mkdirTFile(unbinnedSplinesDirectory, binName)->cd();
            for( size_t iEntry = 0 ; iEntry < graphsListHolder[iBin].size() ; iEntry++ ){

                bool isFlat = false;

                // CHECK IF FLAT
                double Ycheck = -1;
                for(int i_pt = 0 ; i_pt < graphsListHolder[iBin][iEntry]->GetN() ; i_pt++){

                    if(Ycheck == -1){
                        Ycheck = graphsListHolder[iBin][iEntry]->GetY()[i_pt];
                    }
                    else if(Ycheck != graphsListHolder[iBin][iEntry]->GetY()[i_pt]){
                        break;
                    }
                    else if(i_pt+1 == graphsListHolder[iBin][iEntry]->GetN()){
//                        LogAlert << xsecSplineName << ": " << currentEntryBin.generateBinName() << " is FLAT." << std::endl;
                        isFlat = true;
                    }

                }

                if(not isFlat){
                    graphsListHolder[iBin][iEntry]->Write(Form("%i_Graph", int(iEntry)));
                }
                else{
                    graphsListHolder[iBin][iEntry]->Write(Form("%i_FLAT_Graph", int(iEntry)));
                }
            }
        }


        TDirectory* binnedSplinesDirectory = GenericToolbox::mkdirTFile(__mapOutSplineTFiles__[xsecSplineName], "BinnedSplinesPlots");
        binnedSplinesDirectory->cd();

        // Hooking the spline bin the TTree
        for(int iSplitVar = 0 ; iSplitVar < int(__listSplitVarNames__.size()) ; iSplitVar++){
            __mapOutSplineTTrees__[xsecSplineName + "_Binned"]->Branch(
                currentEntryBin.splitVarNameList[iSplitVar].c_str(),
                &currentEntryBin.splitVarValueList[iSplitVar] );
        }
        __mapOutSplineTTrees__[xsecSplineName + "_Binned"]->Branch("kinematicBin", &currentEntryBin.kinematicBin);
        __mapOutSplineTTrees__[xsecSplineName + "_Binned"]->Branch("graph",  &currentEntryBin.graphHandler  );
        __mapOutSplineTTrees__[xsecSplineName + "_Binned"]->Branch("spline", &currentEntryBin.splinePtr);

        for( int iBin = 0 ; iBin < int(processedBins.size()) ; iBin++ ){

            currentEntryBin = processedBins[iBin];
            __mapOutSplineTTrees__[xsecSplineName + "_Binned"]->Fill();
            currentEntryBin.graphHandler->Write((currentEntryBin.generateBinName() + "_TGraph").c_str());

        } // iBin

        __mapOutSplineTFiles__[xsecSplineName]->cd();
        __mapOutSplineTFiles__[xsecSplineName]->WriteObject(__mapOutSplineTTrees__[xsecSplineName+"_Binned"], "BinnedSplines");
        __mapOutSplineTFiles__[xsecSplineName]->Close();
        delete __mapOutSplineTFiles__[xsecSplineName];
        __mapOutSplineTFiles__[xsecSplineName] = nullptr;

        // Freeing up memory
        for( int iBin = 0 ; iBin < int(processedBins.size()) ; iBin++ ){

            delete processedBins[iBin].graphHandler;
            delete processedBins[iBin].splinePtr;

        }

        LogDebug << "RAM loss after " << xsecSplineName
                 << GenericToolbox::parseSizeUnits(GenericToolbox::getProcessMemoryUsageDiffSinceLastCall())
                 << std::endl;

    } // Xsec


}
void processInterpolation(){

    LogWarning << "Interpolating splines on kinematic bins..." << std::endl;

    LogDebug << "RAM: " << GenericToolbox::parseSizeUnits(GenericToolbox::getProcessMemoryUsage()) << std::endl;

    int nbXsecSplines = __listSystematicXsecSplineNames__.size();
    for( int iXsec = 0 ; iXsec < nbXsecSplines ; iXsec++ ){

        std::string xsecSplineName = __listSystematicXsecSplineNames__[iXsec];

        GenericToolbox::displayProgressBar(
            iXsec, nbXsecSplines,
            LogInfo.getPrefixString() + " > Interpolating splines: " + xsecSplineName,
            true);
        LogInfo.quietLineJump();

        // Holders
        std::map<std::string, std::vector<SplineBin>> splittedBinsMap;
        TGraph* graphBuffer = nullptr;
        TSpline3* splineBuffer = nullptr;
        SplineBin currentEntryBin;
        for(int iSplitVar = 0 ; iSplitVar < int(__listSplitVarNames__.size()) ; iSplitVar++){
            currentEntryBin.addSplitVar(__listSplitVarNames__[iSplitVar], -1);
        }
        SplineBin lastEntryBin; lastEntryBin.reset();

        __mapOutSplineTFiles__[xsecSplineName] = TFile::Open(
            Form("%s_splines.root", xsecSplineName.c_str()),
            "READ"
        );
        __mapOutSplineTTrees__[xsecSplineName + "_Binned"] = (TTree*) __mapOutSplineTFiles__[xsecSplineName]->Get("BinnedSplines");
        if(__mapOutSplineTTrees__[xsecSplineName + "_Binned"] == nullptr){
            LogError << "No spline has been found for: " << xsecSplineName << std::endl;
            __mapOutSplineTFiles__[xsecSplineName]->Close();
            delete __mapOutSplineTFiles__[xsecSplineName];
            __mapOutSplineTFiles__[xsecSplineName] = nullptr;
            continue;
        }
        int nbEntries = __mapOutSplineTTrees__[xsecSplineName + "_Binned"]->GetEntries();

        // Hooking the spline bin the TTree
        __mapOutSplineTTrees__[xsecSplineName + "_Binned"]->SetBranchAddress("kinematicBin", &currentEntryBin.kinematicBin);
        for(int iSplitVar = 0 ; iSplitVar < int(__listSplitVarNames__.size()) ; iSplitVar++){
            __mapOutSplineTTrees__[xsecSplineName + "_Binned"]->SetBranchAddress(
                currentEntryBin.splitVarNameList[iSplitVar].c_str(),
                &currentEntryBin.splitVarValueList[iSplitVar]
                );
        }
        __mapOutSplineTTrees__[xsecSplineName + "_Binned"]->SetBranchAddress("graph", &currentEntryBin.graphHandler);
        __mapOutSplineTTrees__[xsecSplineName + "_Binned"]->SetBranchAddress("spline", &currentEntryBin.splinePtr);

        // Reading the tree
        std::string splitBinName;
        for( int iSplineEntry = 0 ; iSplineEntry < nbEntries ; iSplineEntry++ ){

            __mapOutSplineTTrees__[xsecSplineName + "_Binned"]->GetEntry(iSplineEntry);
            splitBinName = "";
            splitBinName = GenericToolbox::joinVectorString(
                GenericToolbox::splitString(currentEntryBin.splinePtr->GetTitle(), "_"),
                "_", 0, -2);

            splittedBinsMap[splitBinName].emplace_back(currentEntryBin);

        }

        // Performing the kinematic bins interpolation
        std::vector<int> interpolatedKinematicBins;
        for( auto& splittedBinPair : splittedBinsMap ){

            // sort kinematic bins
            auto aGoesFirst = []( SplineBin& a, SplineBin&b ){
                return a.kinematicBin < b.kinematicBin;
            };
            std::sort( splittedBinPair.second.begin(), splittedBinPair.second.end(), aGoesFirst );

            LogWarning << "Slice " << splittedBinPair.first << ": "
                    << 100*(1 - double(splittedBinPair.second.size())/double(__listKinematicBins__.size()))
                    << "\% of the bins will be interpolated..."
                    << std::endl;

            int anchorBinIndex = 0;
            std::vector<SplineBin> interpolatedBins;
            for(int iBin = 0 ; iBin < int(__listKinematicBins__.size()) ; iBin++ ){

                if(    anchorBinIndex+1 < splittedBinPair.second.size() // for the last anchorBinIndex this will be false
                   and iBin == splittedBinPair.second[anchorBinIndex].kinematicBin){
                    anchorBinIndex++;
                }
                else {
                    interpolatedBins.emplace_back(splittedBinPair.second[anchorBinIndex]);
                    interpolatedBins.back().kinematicBin = iBin;
                    interpolatedKinematicBins.emplace_back(interpolatedBins.back().kinematicBin);
                }

            }

            // Append the new interpolated bins to the list
            splittedBinPair.second.insert(
                splittedBinPair.second.end(),
                interpolatedBins.begin(),
                interpolatedBins.end()
                );

            // Sort again
            std::sort( splittedBinPair.second.begin(), splittedBinPair.second.end(), aGoesFirst );


        } // splitBin

        // Reopening
        __mapOutSplineTFiles__[xsecSplineName]->Close();
        delete __mapOutSplineTFiles__[xsecSplineName];
        __mapOutSplineTFiles__[xsecSplineName] = TFile::Open(
            Form("%s_splines.root", xsecSplineName.c_str()),
            "UPDATE"
        );
        __mapOutSplineTTrees__[xsecSplineName + "_InterpolatedBinned"] = new TTree("InterpolatedBinnedSplines", "InterpolatedBinnedSplines");

        // Hooking the spline bin the TTree
        for(int iSplitVar = 0 ; iSplitVar < int(__listSplitVarNames__.size()) ; iSplitVar++){
            __mapOutSplineTTrees__[xsecSplineName + "_InterpolatedBinned"]->Branch(
                currentEntryBin.splitVarNameList[iSplitVar].c_str(),
                &currentEntryBin.splitVarValueList[iSplitVar] );
        }
        for( auto& splitVarName: __listSplitVarNames__ ){

        }
        __mapOutSplineTTrees__[xsecSplineName + "_InterpolatedBinned"]->Branch("kinematicBin", &currentEntryBin.kinematicBin);
        __mapOutSplineTTrees__[xsecSplineName + "_InterpolatedBinned"]->Branch("graph",  &currentEntryBin.graphHandler  );
        __mapOutSplineTTrees__[xsecSplineName + "_InterpolatedBinned"]->Branch("spline", &currentEntryBin.splinePtr);

        // Writing
        for( auto& splittedBinPair : splittedBinsMap ){

            for( int iBin = 0 ; iBin < int(splittedBinPair.second.size()) ; iBin++ ){

                currentEntryBin = splittedBinPair.second[iBin];

                std::string title = splittedBinPair.first;
                title += "_" + std::to_string(splittedBinPair.second[iBin].kinematicBin);

                // is it an interpolated bin ?
                if(GenericToolbox::doesElementIsInVector(currentEntryBin.kinematicBin, interpolatedKinematicBins)){
                    title += "_interpolated";
                }

                currentEntryBin.graphHandler->SetTitle(title.c_str());
                currentEntryBin.graphHandler->SetName(title.c_str());
                currentEntryBin.splinePtr->SetTitle(title.c_str());
                currentEntryBin.splinePtr->SetName(title.c_str());
                __mapOutSplineTTrees__[xsecSplineName + "_InterpolatedBinned"]->Fill();

            } // iBin

        } // splitBin

        __mapOutSplineTFiles__[xsecSplineName]->WriteObject(
            __mapOutSplineTTrees__[xsecSplineName+"_InterpolatedBinned"], "InterpolatedBinnedSplines"
        );
        __mapOutSplineTFiles__[xsecSplineName]->Close();
        delete __mapOutSplineTFiles__[xsecSplineName];
        __mapOutSplineTFiles__[xsecSplineName] = nullptr;

        // Freeing memory
        std::vector<TGraph*> deletedGraphs;
        std::vector<TSpline3*> deletedSplines;
        for( auto& splittedBinPair : splittedBinsMap ){

            for( int iBin = 0 ; iBin < int(splittedBinPair.second.size()) ; iBin++ ){

                if(not GenericToolbox::doesElementIsInVector(splittedBinPair.second[iBin].graphHandler, deletedGraphs)){
                    deletedGraphs.emplace_back(splittedBinPair.second[iBin].graphHandler);
                    delete splittedBinPair.second[iBin].graphHandler;
                }

                if(not GenericToolbox::doesElementIsInVector(splittedBinPair.second[iBin].splinePtr, deletedSplines)){
                    deletedSplines.emplace_back(splittedBinPair.second[iBin].splinePtr);
                    delete splittedBinPair.second[iBin].splinePtr;
                }


            } // iBin

        }

    }

    LogDebug << "RAM: " << GenericToolbox::parseSizeUnits(GenericToolbox::getProcessMemoryUsage()) << std::endl;
    LogDebug << "Max RAM: " << GenericToolbox::parseSizeUnits(GenericToolbox::getProcessMaxMemoryUsage()) << std::endl;

}

bool buildTreeSyncCache(){

    if(__treeConvEntryToVertexIDSplit__.empty()) mapTreeConverterEntries();

    LogInfo << "Building tree sync cache..." << std::endl;

    // genWeights tree (same number of events as __flattree__)
    int nbGenWeightsEntries = __currentSampleSumTTree__->GetEntries();

    Int_t sTrueVertexID;
    Int_t sRun;
    Int_t sSubRun;

    __currentFlatTree__->SetBranchStatus("*", false);

    __currentFlatTree__->SetBranchStatus("sTrueVertexID", true);
    __currentFlatTree__->SetBranchStatus("sRun", true);
    __currentFlatTree__->SetBranchStatus("sSubRun", true);

//    __currentFlatTree__->SetBranchAddress("sTrueVertexID[0]", &sTrueVertexID); // Can't do that
    __currentFlatTree__->SetBranchAddress("sRun", &sRun);
    __currentFlatTree__->SetBranchAddress("sSubRun", &sSubRun);

    __currentGenWeightToTreeConvEntry__.clear();

    int nbMatches = 0, nbMissing = 0;
    for(int iGenWeightsEntry = 0 ; iGenWeightsEntry < nbGenWeightsEntries; iGenWeightsEntry++){ // genWeights
        __currentFlatTree__->GetEntry(iGenWeightsEntry);
        __currentGenWeightToTreeConvEntry__[iGenWeightsEntry] = -1; // reset

        int tcEntry = -1;
        int tcVertexID = -1;
        for(const auto& runMapPair : __treeConvEntryToVertexIDSplit__){
            if(runMapPair.first != sRun) continue;
            for(const auto& subrunMapPair : runMapPair.second){
                if(subrunMapPair.first != sSubRun) continue;

                sTrueVertexID = __currentFlatTree__->GetLeaf("sTrueVertexID")->GetValue(0);

                for(const auto& tcEntryToVertexID : subrunMapPair.second){
                    tcEntry = tcEntryToVertexID.first;
                    tcVertexID = tcEntryToVertexID.second;

//                    if( tcEntry < __lastTCEntryFound__ ) continue; // all events should be in order

                    // Looking for the sTrueVertexID in TC
                    if(tcVertexID != sTrueVertexID ) {
                        continue;
                    }
                    else{
                        // Check if the corresponding run/subrun matches
                        __currentGenWeightToTreeConvEntry__[iGenWeightsEntry] = tcEntry;
                        __lastTCEntryFound__                                  = tcEntry;
                        break;

                    }
                }
            }
        }
        if(__currentGenWeightToTreeConvEntry__[iGenWeightsEntry] == -1){
            nbMissing++;
        }
        else {
            nbMatches++;
        }
    }

    LogInfo << "Tree synchronization has: " << nbMatches << " matches, " << nbMissing << " miss." << std::endl;

    __currentFlatTree__->ResetBranchAddress(__currentFlatTree__->GetBranch("sTrueVertexID"));
    __currentFlatTree__->ResetBranchAddress(__currentFlatTree__->GetBranch("sRun"));
    __currentFlatTree__->ResetBranchAddress(__currentFlatTree__->GetBranch("sSubRun"));

    __treeConverterRecoTTree__->ResetBranchAddress(__treeConverterRecoTTree__->GetBranch("run"));
    __treeConverterRecoTTree__->ResetBranchAddress(__treeConverterRecoTTree__->GetBranch("subrun"));

    __currentFlatTree__->SetBranchStatus("*", true);

    if(nbMatches == 0){
        LogError << "Could not sync genWeights file and TreeConverter." << std::endl;
        return false;
    }
    return true;

}
void mapTreeConverterEntries(){

    LogInfo << "Mapping Tree Converter File..." << std::endl;

    Int_t vertexID;
    int run;
    int subrun;
    __treeConverterRecoTTree__->SetBranchAddress("vertexID", &vertexID);
    __treeConverterRecoTTree__->SetBranchAddress("run", &run);
    __treeConverterRecoTTree__->SetBranchAddress("subrun", &subrun);

    std::string loadTitle = LogInfo.getPrefixString() + "Mapping TC entries...";
    for(int iEntry = 0 ; iEntry < __treeConverterRecoTTree__->GetEntries(); iEntry++){
        GenericToolbox::displayProgressBar(iEntry, __treeConverterRecoTTree__->GetEntries(), loadTitle);
        __treeConverterRecoTTree__->GetEntry(iEntry);
        __treeConvEntryToVertexIDSplit__[run][subrun][iEntry] = vertexID;
    }
    __treeConverterRecoTTree__->ResetBranchAddress(__treeConverterRecoTTree__->GetBranch("vertexID"));

}
void fillSplineBin(SplineBin& splineBin_, TTree* tree_){

    if(tree_ == nullptr) return;

    splineBin_.D1Reco = tree_->GetLeaf("D1Reco")->GetValue(0);
    splineBin_.D2Reco = tree_->GetLeaf("D2Reco")->GetValue(0);

    for(int iSplitVar = 0 ; iSplitVar < int(__listSplitVarNames__.size()) ; iSplitVar++){
        splineBin_.splitVarValueList[iSplitVar] = tree_->GetLeaf(splineBin_.splitVarNameList[iSplitVar].c_str())->GetValue(0);
    }

    splineBin_.kinematicBin = -1;
    for(int iBin = 0 ; iBin < __listKinematicBins__.size() ; iBin++){
        if(
            splineBin_.D1Reco >= __listKinematicBins__[iBin].D1low
            and splineBin_.D1Reco < __listKinematicBins__[iBin].D1high
            and splineBin_.D2Reco >= __listKinematicBins__[iBin].D2low
            and splineBin_.D2Reco < __listKinematicBins__[iBin].D2high
            ){
            splineBin_.kinematicBin = iBin;
            break;
        }
    }

}

// Config Generation
void generateJsonConfigFile(){

    LogInfo << "Generating json config file..." << std::endl;

    if(__pathCovarianceMatrixFile__.empty()){
        LogError << "__covariance_matrix_file_path__ is not set." << std::endl;
        exit(EXIT_FAILURE);
    }

    TFile* covariance_matrix_tfile = TFile::Open(__pathCovarianceMatrixFile__.c_str(), "READ");
    if(not covariance_matrix_tfile->IsOpen()){
        LogError << "Could not open : " << covariance_matrix_tfile->GetName() << std::endl;
        exit(EXIT_FAILURE);
    }

    auto* xsec_param_names = dynamic_cast<TObjArray *>(covariance_matrix_tfile->Get("xsec_param_names"));
    auto* xsec_param_nom_unnorm = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_nom_unnorm"));
    auto* xsec_param_prior_unnorm = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_prior_unnorm"));
    auto* xsec_param_lb = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_lb"));
    auto* xsec_param_ub = (TVectorT<double> *)(covariance_matrix_tfile->Get("xsec_param_ub"));

    std::vector<std::string> dial_strings;

    for(int iSpline = 0 ; iSpline < xsec_param_names->GetEntries() ; iSpline++ ){

        std::string xsecSplineName = xsec_param_names->At(iSpline)->GetName();

        double step;
        step = std::min(
            fabs((*xsec_param_nom_unnorm)[iSpline] - (*xsec_param_ub)[iSpline]),
            fabs((*xsec_param_nom_unnorm)[iSpline] - (*xsec_param_lb)[iSpline])
        );
        step *= 0.05;

        if(GenericToolbox::doesElementIsInVector(xsecSplineName, __listSystematicXsecSplineNames__)){

            // Splined parameter
            std::stringstream dial_ss;
            dial_ss << "    {" << std::endl;
            dial_ss << "      \"name\" : \"" << xsecSplineName << "\"," << std::endl;
            dial_ss << "      \"is_normalization_dial\" : false," << std::endl;
            dial_ss << "      \"nominal\" : " << (*xsec_param_nom_unnorm)[iSpline] << "," << std::endl;
            dial_ss << "      \"prior\" : " << (*xsec_param_prior_unnorm)[iSpline] << "," << std::endl;
            dial_ss << "      \"step\" : " << step << "," << std::endl;
            dial_ss << "      \"limit_lo\" : " << (*xsec_param_lb)[iSpline] << "," << std::endl;
            dial_ss << "      \"limit_hi\" : " << (*xsec_param_ub)[iSpline] << "," << std::endl;
            dial_ss << "      \"binning\" : \"" << __pathBinningFile__ << "\"," << std::endl;
            dial_ss << "      \"splines\" : \"" << Form("%s_splines.root", xsecSplineName.c_str())  << "\"," << std::endl;
            dial_ss << "      \"use\" : true" << std::endl; // The fitter will throw us out if no splines are to be found
            dial_ss << "    }";

            dial_strings.emplace_back(dial_ss.str());

        }
        else{

            // Normalization parameter
            std::stringstream dial_ss;
            dial_ss << "    {" << std::endl;
            dial_ss << "      \"name\" : \"" << xsecSplineName << "\"," << std::endl;
            dial_ss << "      \"is_normalization_dial\" : true," << std::endl;
            dial_ss << "      \"nominal\" : " << (*xsec_param_nom_unnorm)[iSpline] << "," << std::endl;
            dial_ss << "      \"prior\" : " << (*xsec_param_prior_unnorm)[iSpline] << "," << std::endl;
            dial_ss << "      \"step\" : " << step << "," << std::endl; // 5 %
            dial_ss << "      \"limit_lo\" : " << (*xsec_param_lb)[iSpline] << "," << std::endl;
            dial_ss << "      \"limit_hi\" : " << (*xsec_param_ub)[iSpline] << "," << std::endl;

            std::map<std::string, std::vector<int>> applyOnlyOnMap;
            std::vector<std::string> applyConditionsList;

            if(     GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "2p2h_norm")){
                // 2p2h_norm* - 2p2h
                applyOnlyOnMap["reaction"].emplace_back(__toReactionIndex__["2p2h"]);
                applyConditionsList.emplace_back("reaction == " + std::to_string(__toReactionIndex__["2p2h"]));
                if(xsecSplineName == "2p2h_norm_nu"){
                    applyConditionsList.emplace_back("nutype > 0");
                }
                else if(xsecSplineName == "2p2h_norm_nubar"){
                    applyConditionsList.emplace_back("nutype < 0");
                }
                else if(xsecSplineName == "2p2h_norm_nubar"){
                    // TODO: implement anti-correlations in XsecDial
//                    applyConditionsList.emplace_back("(target == 6)*(1)"); // C
//                    applyConditionsList.emplace_back("(target == 8)*(-1)"); // O
                }
            }
            if(   GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "Q2_")
//               or GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "EB_") // EB dials are described by splines
               or GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "CC_norm_")
               ){
                applyOnlyOnMap["reaction"].emplace_back(__toReactionIndex__["CCQE"]);
                applyConditionsList.emplace_back("reaction == " + std::to_string(__toReactionIndex__["CCQE"]));
            }

            // TN344, Table 1
            if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "Q2_norm_0")){
                applyConditionsList.emplace_back("q2_true > 0");
                applyConditionsList.emplace_back("q2_true < 0.05*1E6"); // q2_true is expressed in keV -> convert to GeV
            }
            else if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "Q2_norm_1")){
                applyConditionsList.emplace_back("q2_true > 0.05*1E6");
                applyConditionsList.emplace_back("q2_true < 0.10*1E6");
            }
            else if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "Q2_norm_2")){
                applyConditionsList.emplace_back("q2_true > 0.10*1E6");
                applyConditionsList.emplace_back("q2_true < 0.15*1E6");
            }
            else if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "Q2_norm_3")){
                applyConditionsList.emplace_back("q2_true > 0.15*1E6");
                applyConditionsList.emplace_back("q2_true < 0.20*1E6");
            }
            else if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "Q2_norm_4")){
                applyConditionsList.emplace_back("q2_true > 0.20*1E6");
                applyConditionsList.emplace_back("q2_true < 0.25*1E6");
            }
            else if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "Q2_norm_5")){
                applyConditionsList.emplace_back("q2_true > 0.25*1E6");
                applyConditionsList.emplace_back("q2_true < 0.50*1E6");
            }
            else if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "Q2_norm_6")){
                applyConditionsList.emplace_back("q2_true > 0.50*1E6");
                applyConditionsList.emplace_back("q2_true < 1.00*1E6");
            }
            else if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "Q2_norm_7")){
                applyConditionsList.emplace_back("q2_true > 1.00*1E6");
            }

            if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "CC_Misc")){
                // CC_Misc - CCGamma, CCKaon, CCEta
                // CC Misc Misc Spline 100% normalisation error on CC1γ, CC1K, CC1η.
                // all reactions but -> only CC-Other
                applyOnlyOnMap["topology"].emplace_back(__toTopologyIndex__["CC-Other"]);
                applyConditionsList.emplace_back("topology == " + std::to_string(__toTopologyIndex__["CC-Other"]));
            }
            if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "CC_DIS_MultPi")){
                // CC_DIS_MultiPi_Norm_Nu - CCDIS and CCMultPi, nu only
                // CC_DIS_MultiPi_Norm_nuBar - CCDIS and CCMultPi, anu only
                applyOnlyOnMap["topology"].emplace_back(__toTopologyIndex__["CC-Other"]);
                applyOnlyOnMap["reaction"].emplace_back(__toReactionIndex__["DIS"]);

                applyConditionsList.emplace_back("topology == " + std::to_string(__toTopologyIndex__["CC-Other"]));
                applyConditionsList.emplace_back("reaction == " + std::to_string(__toReactionIndex__["DIS"]));
                if(xsecSplineName == "CC_DIS_MultPi_Norm_Nu"){
                    applyConditionsList.emplace_back("nutype > 0");
                }
                else if(xsecSplineName == "CC_DIS_MultPi_Norm_Nubar"){
                    applyConditionsList.emplace_back("nutype < 0");
                }
            }
            if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "CC_Coh_")){
                //
                applyOnlyOnMap["topology"].emplace_back(__toTopologyIndex__["CC-Other"]);
                applyOnlyOnMap["reaction"].emplace_back(__toReactionIndex__["Coh"]);
//                applyConditionsList.emplace_back("topology == " + std::to_string(__toTopologyIndex__["CC-Other"]));
                applyConditionsList.emplace_back("reaction == " + std::to_string(__toReactionIndex__["Coh"]));
                if(xsecSplineName == "CC_Coh_O"){
                    applyConditionsList.emplace_back("target == 8");
                }
                if(xsecSplineName == "CC_Coh_C"){
                    applyConditionsList.emplace_back("target == 6");
                }
            }

            if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "nue_numu")){
                // nue_numu - All nue
                applyOnlyOnMap["nutype"].emplace_back(12);
                // TODO: CHECK? my guess: Implement anti-correlated dials
                applyConditionsList.emplace_back("(nutype == 12)*(1)");
//                applyConditionsList.emplace_back("(nutype == 14)*(-1)");
            }
            else if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "nuebar_numubar")){
                // nuebar_numubar - All nuebar
                // TODO: CHECK? my guess: Implement anti-correlated dials
                applyConditionsList.emplace_back("nutype == -12");
//                applyConditionsList.emplace_back("(nutype == -14)*(-1)");
            }

            if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "NC_Coh")){
                applyConditionsList.emplace_back("reaction == " + std::to_string(__toReactionIndex__["Coh"]));
            }
            else if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "NC")){
                applyConditionsList.emplace_back("reaction == " + std::to_string(__toReactionIndex__["NC"]));
            }

            if(not applyOnlyOnMap.empty() and false){ // DISABLED
                dial_ss << "      \"apply_only_on\" : {" << std::endl;
                bool isFirstCondition = true;
                for(const auto& applyOnlyOnCondition : applyOnlyOnMap){
                    if(not isFirstCondition) dial_ss << "," << std::endl;
                    isFirstCondition = false;
                    dial_ss << "        \"" << applyOnlyOnCondition.first << "\" : [ ";
                    for(size_t iInt = 0 ; iInt < applyOnlyOnCondition.second.size() ; iInt++){
                        if(iInt != 0) dial_ss << ", ";
                        dial_ss << applyOnlyOnCondition.second[iInt];
                    }
                    dial_ss << " ]";
                }
                dial_ss << std::endl << "      }," << std::endl;
            }

            std::map<std::string, std::vector<int>> dontApplyOnMap;

            if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "NC_other_near")){
                // NC_other_near - all NC which isn't Coh or 1gamma
                dontApplyOnMap["reaction"].emplace_back(__toReactionIndex__["Coh"]);
//                applyConditionsList.emplace_back("reaction != " + std::to_string(__toReactionIndex__["Coh"]));
            }

            if(not dontApplyOnMap.empty() and false){ // DISABLED
                dial_ss << "      \"dont_apply_on\" : {" << std::endl;
                bool isFirstCondition = true;
                for(const auto& dontApplyOnCondition : dontApplyOnMap){
                    if(not isFirstCondition) dial_ss << "," << std::endl;
                    isFirstCondition = false;
                    dial_ss << "        \"" << dontApplyOnCondition.first << "\" : [ ";
                    for(size_t iInt = 0 ; iInt < dontApplyOnCondition.second.size() ; iInt++){
                        if(iInt != 0) dial_ss << ", ";
                        dial_ss << dontApplyOnCondition.second[iInt];
                    }
                    dial_ss << " ]" << std::endl;
                }
                dial_ss << "      }," << std::endl;
            }

            if(not applyConditionsList.empty()){
                dial_ss << "      \"apply_condition\" : \"" << GenericToolbox::joinVectorString(applyConditionsList, " && ") << "\"," << std::endl;
            }
            else{
                LogError << "No apply condition for normalization spline: " << xsecSplineName << std::endl;
            }

            if(GenericToolbox::doesStringStartsWithSubstring(xsecSplineName, "EB")) dial_ss << "      \"use\" : false" << std::endl; // DISABLED
            else dial_ss << "      \"use\" : true" << std::endl;
            dial_ss << "    }";




            dial_strings.emplace_back(dial_ss.str());

        }

    }

    std::vector<std::string> input_dir_elements;
    input_dir_elements.emplace_back("inputs");
    input_dir_elements.emplace_back("BANFF_Fit");
    auto cwd = GenericToolbox::getCurrentWorkingDirectory();
    auto path_elements = GenericToolbox::splitString(cwd, "/");
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
    config_ss << "  \"input_dir\" : \"/" << GenericToolbox::joinVectorString(input_dir_elements, "/") << "/\"," << std::endl;
    config_ss << "  \"dials\" :" << std::endl;
    config_ss << "  [" << std::endl;
    config_ss << GenericToolbox::joinVectorString(dial_strings, ",\n") << std::endl;
    config_ss << "  ]" << std::endl;
    config_ss << "}" << std::endl;
    GenericToolbox::dumpStringInFile("./config_splines.json", config_ss.str());

    covariance_matrix_tfile->Close();

}

double convertToAbsoluteVariation(std::string parName_, double relativeDeviationParamValue_){

    double output;
    if(doesParamHasRelativeXScale(parName_)){
        // 0 -> nominal value
        // 0.1 -> +10% of the initial parameter
        output = __mapNominalValues__[parName_]*(1 + relativeDeviationParamValue_);
    }
    else{
        // 0 -> nominal value
        // 0.1 -> +10% of 1 sigma
        output = __mapNominalValues__[parName_] + relativeDeviationParamValue_ * __mapErrorValues__[parName_];
    }
    return output;
}
bool doesParamHasRelativeXScale(std::string& systName_){
    bool is_relative_X = false;
    std::string genWeigths_name = __mapXsecToGenWeights__[systName_];
    if(genWeigths_name.empty()){
        genWeigths_name = systName_;
    }
    for(auto& relative_component_name: __listRelativeVariationComponent__){

        if(GenericToolbox::doesStringContainsSubstring(
            GenericToolbox::toLowerCase(relative_component_name),
            GenericToolbox::toLowerCase(genWeigths_name)
        )
            ){
//            std::cout << GenericToolbox::to_lower_case(relative_component_name) << " has ";
//            std::cout << GenericToolbox::to_lower_case(genWeigths_name) << " / " << systName_ << std::endl;
            is_relative_X = true;
            break;
        }
    }
    return is_relative_X;
}
