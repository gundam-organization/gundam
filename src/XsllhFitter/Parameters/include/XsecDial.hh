#ifndef XSECDIAL_H
#define XSECDIAL_H

// STD
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ROOT
#include "TTree.h"
#include <TCollection.h>
#include <TFile.h>
#include <TGraph.h>
#include <TKey.h>
#include <TSpline.h>
#include <TTreeFormula.h>

// Submodules
#include "GenericToolbox.h"

// This Project
#include "AnaEvent.hh"
#include "BinManager.hh"
#include "ColorOutput.hh"
#include "GundamGlobals.h"

struct SplineBin {

    std::vector<std::string> splitVarNameList{};
    std::vector<int>         splitVarValueList{};
    int kinematicBin = -1;

    int entry = -1;

    Float_t D1Reco = -1;
    Float_t D2Reco = -1;
    TGraph* graphHandler = nullptr;
    TSpline3* splinePtr  = nullptr;

    std::string print(){
        std::stringstream out;
        out << GET_VAR_NAME_VALUE(entry) << std::endl;
        out << GET_VAR_NAME_VALUE(D1Reco) << std::endl;
        out << GET_VAR_NAME_VALUE(D2Reco) << std::endl;
        out << GET_VAR_NAME_VALUE(graphHandler) << std::endl;
        out << GET_VAR_NAME_VALUE(splinePtr) << std::endl;
        for( int iSplitVar = 0 ; iSplitVar < int(splitVarNameList.size()) ; iSplitVar++ ){
            out << splitVarNameList[iSplitVar] << " = " << splitVarValueList[iSplitVar] << std::endl;
        }
        out << GET_VAR_NAME_VALUE(kinematicBin) << std::endl;
        return out.str();
    }
    std::string generateBinName() const {
        std::stringstream out;
        for( int iSplitVar = 0 ; iSplitVar < int(splitVarNameList.size()) ; iSplitVar++ ){
            out << splitVarNameList[iSplitVar] << "_" << splitVarValueList[iSplitVar] << "_";
        }
        out << "bin_" << kinematicBin;
        return out.str();
    }
    int getSplitVarValue(const std::string& varName_){
        for( int iSplitVar = 0 ; iSplitVar < int(splitVarNameList.size()) ; iSplitVar++ ){
            if( splitVarNameList[iSplitVar] == varName_ ) return splitVarValueList[iSplitVar];
        }
        throw std::logic_error("NOT FOUND");
        return -1;
    }
    void addSplitVar(std::string splitVarName_, int splitValue_){
        splitVarNameList.emplace_back(splitVarName_);
        splitVarValueList.emplace_back(splitValue_);
    }
    void reset(){
        entry = -1;
        D1Reco = -1;
        D2Reco = -1;
        kinematicBin = -1;
        for( int iSplitVar = 0 ; iSplitVar < int(splitVarNameList.size()) ; iSplitVar++ ){
            splitVarValueList[iSplitVar] = -1;
        }
        graphHandler = nullptr;
        splinePtr    = nullptr;
    }
    bool operator==(const SplineBin& otherBin){
        if(this->kinematicBin != otherBin.kinematicBin) return false;
        for( int iSplitVar = 0 ; iSplitVar < int(splitVarNameList.size()) ; iSplitVar++ ){
            int otherBinIndex = GenericToolbox::findElementIndex(splitVarNameList[iSplitVar], otherBin.splitVarNameList);
            if( otherBinIndex == -1 ) return false;
            if( this->splitVarValueList.at(iSplitVar) != otherBin.splitVarValueList.at(otherBinIndex) ){
                return false;
            }
        }
        return true;
    }
    SplineBin& operator=(const SplineBin& otherBin){

        this->splitVarNameList.clear();
        this->splitVarValueList.clear();
        reset();

        for( int iSplitVar = 0 ; iSplitVar < int(otherBin.splitVarNameList.size()) ; iSplitVar++ ){
            this->splitVarNameList.emplace_back(otherBin.splitVarNameList.at(iSplitVar));
            this->splitVarValueList.emplace_back(otherBin.splitVarValueList.at(iSplitVar));
        }

        kinematicBin = otherBin.kinematicBin;
        entry = otherBin.entry;
        D1Reco = otherBin.D1Reco;
        D2Reco = otherBin.D2Reco;
        graphHandler = otherBin.graphHandler;
        splinePtr    = otherBin.splinePtr;
        return *this;
    }
};

class XsecDial
{
public:

    explicit XsecDial(std::string  dial_name);
    XsecDial(std::string  dial_name, const std::string& fname_splines,
             const std::string& fname_binning);

    void SetBinning(const std::string& fname_binning);
    void SetApplyOnlyOnMap(const std::map<std::string, std::vector<int>>& applyOnlyOnMap_);
    void SetDontApplyOnMap(const std::map<std::string, std::vector<int>>& dontApplyOnMap_);
    void SetApplyCondition(std::string applyCondition_);
    void ReadSplines(const std::string& fname_splines);

    bool GetIsSplinesInTree() const;
    bool GetIsNormalizationDial() const;

    int GetSplineIndex(const std::vector<int>& var, const std::vector<double>& bin) const;
    int GetSplineIndex(AnaEvent* eventPtr_, SplineBin* eventSplineBinPtr_ = nullptr);
    int GetSplineIndex(SplineBin& splineBinToLookFor_);
    double GetSplineValue(int index, double dial_value) const;
    double GetBoundedValue(int splineIndex_, double dialValue_);
    std::string GetSplineName(int index) const;
    std::map<std::string, std::vector<int>>* GetApplyOnlyOnMapPtr() { return &_applyOnlyOnMap_; };
    std::map<std::string, std::vector<int>>* GetDontApplyOnMapPtr() { return &_dontApplyOnMap_; };
    const std::string& GetApplyCondition() const;
    std::vector<TTreeFormula*>& getApplyConditionFormulaeList();

    std::vector<TSpline3*>& GetSplinePtrList(){ return _splinePtrList_; }
    std::vector<TSpline3>& GetSplineList(){ return v_splines; }

    void SetVars(double nominal, double step, double limit_lo, double limit_hi);
    void SetDimensions(const std::vector<int>& dim);
    void Print(bool print_bins = false) const;

    void SetNominal(double nominal);
    void SetPrior(double prior);
    void SetStep(double step);
    void SetLimitLo(double limit_lo);
    void SetLimitHi(double limit_hi);
    void SetIsNormalizationDial(bool isNormalizationDial_);

    std::string GetName() const { return m_name; }
    double GetNominal() const { return m_nominalDial; }
    double GetPrior() const { return m_prior; }
    double GetStep() const { return m_step; }
    double GetLimitLow() const { return m_limit_lo; }
    double GetLimitHigh() const { return m_limit_hi; }

    TSpline3* getCorrespondingSpline(AnaEvent* anaEvent_);

private:

    unsigned int nbins{};
    double m_nominalDial{};
    double m_prior{};
    double m_step{};
    double m_limit_lo{};
    double m_limit_hi{};

    std::string m_name;

    bool _useGraphEval_;

    //std::vector<TGraph> v_graphs;
    std::vector<SplineBin> _splineBinList_;
    std::vector<std::string> v_splitVarNames;
    std::vector<TSpline3> v_splines;
    std::vector<TSpline3*> _splinePtrList_;
    std::vector<TGraph*> _graphPtrList_;
    std::vector<std::pair<double,double>> _graphBoundsList_;
    std::vector<std::string> _splineNameList_;
    std::vector<int> m_dimensions;

    BinManager bin_manager;

    bool _isNormalizationDial_;
    bool _isSplinesInTTree_;
    TTree* _interpolatedBinnedSplinesTTree_{};

    SplineBin _splineBinBuffer_;
    std::vector<std::vector<std::pair<double, double>>> _binEdgesList_;
    std::map<std::string, std::vector<int>> _applyOnlyOnMap_;
    std::map<std::string, std::vector<int>> _dontApplyOnMap_;
    std::string _applyCondition_;

    std::vector<TTreeFormula*> _applyConditionFormulaeList_; // threads

    typedef struct{
        bool isBeingEdited;
        double cachedDialValue;
        double cachedDialWeight;
    } SplineCache;

    std::vector<SplineCache> _splineCacheList_;

    const std::string TAG = color::MAGENTA_STR + "[XsecDial]: " + color::RESET_STR;
};

#endif
