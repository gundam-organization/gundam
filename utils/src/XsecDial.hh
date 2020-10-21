#ifndef XSECDIAL_H
#define XSECDIAL_H

// STD
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ROOT
#include <TCollection.h>
#include <TFile.h>
#include <TGraph.h>
#include <TKey.h>
#include <TSpline.h>
#include "TTree.h"

// Submodules
#include "GenericToolbox.h"

// This Project
#include "../../anaevents/src/AnaEvent.hh"
#include "BinManager.hh"
#include "ColorOutput.hh"

struct SplineBin{
    std::map<std::string, int> splitVarValue{};
    int kinematicBin = -1;
    Float_t D1Reco = -1;
    Float_t D2Reco = -1;
    TGraph* graphHandler = nullptr;
    TSpline3* splineHandler = nullptr;

    std::string print(){
        std::stringstream out;
        out << GET_VAR_NAME_VALUE(D1Reco) << std::endl;
        out << GET_VAR_NAME_VALUE(D2Reco) << std::endl;
        out << GET_VAR_NAME_VALUE(graphHandler) << std::endl;
        out << GET_VAR_NAME_VALUE(splineHandler) << std::endl;
        for(const auto& splitVarVal : splitVarValue){
            out << splitVarVal.first << " = " << splitVarVal.second << std::endl;
        }
        out << GET_VAR_NAME_VALUE(kinematicBin) << std::endl;
        return out.str();
    }
    std::string getBinName(){
        std::stringstream out;
        for(const auto& splitVarVal : splitVarValue){
            out << splitVarVal.first << "_" << splitVarVal.second << "_";
        }
        out << "bin_" << kinematicBin;
        return out.str();
    }
    void reset(){
        D1Reco = -1;
        D2Reco = -1;
        kinematicBin = -1;
        for(auto& splitVarPair : splitVarValue){
            splitVarPair.second = -1;
        }
        graphHandler = nullptr;
        splineHandler = nullptr;
    }
    bool operator==(const SplineBin& otherBin){
        if(this->kinematicBin != otherBin.kinematicBin) return false;
        for(const auto& splitVarVal : this->splitVarValue){
            if(otherBin.splitVarValue.count( splitVarVal.first ) == 0) return false;
            if(this->splitVarValue[splitVarVal.first] != otherBin.splitVarValue.at(splitVarVal.first)){
                return false;
            }
        }
        return true;
    }
    SplineBin& operator=(const SplineBin& otherBin){
        for(const auto& splitVar : otherBin.splitVarValue){
            splitVarValue[splitVar.first] = otherBin.splitVarValue.at(splitVar.first);
        }
        kinematicBin = otherBin.kinematicBin;
        D1Reco = otherBin.D1Reco;
        D2Reco = otherBin.D2Reco;
        graphHandler = otherBin.graphHandler;
        splineHandler = otherBin.splineHandler;
        return *this;
    }
};

class XsecDial
{
    public:

        explicit XsecDial(const std::string& dial_name);
        XsecDial(const std::string& dial_name, const std::string& fname_splines,
                 const std::string& fname_binning);

        void SetBinning(const std::string& fname_binning);
        void ReadSplines(const std::string& fname_splines);

        bool GetUseSplineSplitMapping() const;

        int GetSplineIndex(const std::vector<int>& var, const std::vector<double>& bin) const;
        int GetSplineIndex(AnaEvent* anaEvent_);
        double GetSplineValue(int index, double dial_value) const;
        double GetBoundedValue(int index, double dial_value) const;
        double GetBoundedValue(AnaEvent* anaEvent_, double dial_value_);
        std::string GetSplineName(int index) const;

        void SetVars(double nominal, double step, double limit_lo, double limit_hi);
        void SetDimensions(const std::vector<int>& dim);
        void Print(bool print_bins = false) const;

        std::string GetName() const { return m_name; }
        double GetNominal() const { return m_nominal; }
        double GetStep() const { return m_step; }
        double GetLimitLow() const { return m_limit_lo; }
        double GetLimitHigh() const { return m_limit_hi; }

        TSpline3* getCorrespondingSpline(AnaEvent* anaEvent_);

    private:
        unsigned int nbins{};
        std::string m_name;
        double m_nominal{};
        double m_step{};
        double m_limit_lo{};
        double m_limit_hi{};
        //std::vector<TGraph> v_graphs;
        std::vector<std::string> v_splitVarNames;
        std::vector<TSpline3> v_splines;
        std::vector<TSpline3*> v_splinesPtr;
        std::vector<std::string> v_splineNames;
        std::vector<int> m_dimensions;
        BinManager bin_manager;

        bool _useSplineSplitMapping_;
        TTree* _interpolatedBinnedSplinesTTree_;
        std::map<std::string, TSpline3*> _splineMapping_;
        SplineBin _splineSplitBinHandler_;
        std::vector<std::vector<std::pair<double, double>>> _binEdgesList_;

        std::map<AnaEvent*, std::pair<bool, TSpline3*> > _eventSplineMapping_;


        const std::string TAG = color::MAGENTA_STR + "[XsecDial]: " + color::RESET_STR;
};

#endif
