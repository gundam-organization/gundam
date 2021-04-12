//
// Created by Adrien BLANCHET on 12/04/2021.
//

#include <stdexcept>

#include "GeneralizedFitBin.h"
#include "Logger.h"

GeneralizedFitBin::GeneralizedFitBin()
{
    Logger::setUserHeaderStr("[GeneralizedFitBin]");
}
GeneralizedFitBin::GeneralizedFitBin(const std::vector<double>& lowBinEdgeList_, const std::vector<double>& highBinEdgeList_)
{
    Logger::setUserHeaderStr("[GeneralizedFitBin]");
    setBinEdges(lowBinEdgeList_, highBinEdgeList_);
}

void GeneralizedFitBin::setBinEdges(const std::vector<double>& lowBinEdgeList_, const std::vector<double>& highBinEdgeList_)
{
    if( lowBinEdgeList_.size() != highBinEdgeList_.size() ){
        LogError << "Low/High bin edges vectors have different sizes." << std::endl;
        throw std::logic_error("Low/High bin edges vectors have different sizes.");
    }

    _lowBinEdgeList_ = lowBinEdgeList_;
    _highBinEdgeList_ = highBinEdgeList_;
}

size_t GeneralizedFitBin::getNbDimensions() const{
    return _lowBinEdgeList_.size();
}
double GeneralizedFitBin::getBinLowEdge(size_t dimensionIndex_) const {
    if( dimensionIndex_ >= _lowBinEdgeList_.size() ){
        LogError << "dimensionIndex_ too high for _lowBinEdgeList_." << std::endl;
        throw std::runtime_error("dimensionIndex_ too high for _lowBinEdgeList_.");
    }
    return _lowBinEdgeList_.at(dimensionIndex_);
}
double GeneralizedFitBin::getBinHighEdge(size_t dimensionIndex_) const {
    if( dimensionIndex_ >= _highBinEdgeList_.size() ){
        LogError << "dimensionIndex_ too high for _highBinEdgeList_." << std::endl;
        throw std::runtime_error("dimensionIndex_ too high for _highBinEdgeList_.");
    }
    return _highBinEdgeList_.at(dimensionIndex_);
}
const std::vector<double>& GeneralizedFitBin::getLowBinEdgeList() const
{
    return _lowBinEdgeList_;
}
const std::vector<double>& GeneralizedFitBin::getHighBinEdgeList() const
{
    return _highBinEdgeList_;
}

bool GeneralizedFitBin::isInBin(const std::vector<double>& eventVarList_) const
{
    if( eventVarList_.size() !=  _lowBinEdgeList_.size() ){
        LogError << "The size of the event var list does not match the dimension of the fit binning." << std::endl;
        throw std::logic_error("The size of the event var list does not match the dimension of the fit binning.");
    }

    bool isCandidate = true;
    for( size_t iVar = 0 ; iVar < eventVarList_.size() ; iVar++ ){
        if(     eventVarList_[iVar] >= this->getBinLowEdge(iVar)
            and eventVarList_[iVar] < this->getBinHighEdge(iVar) ){
            continue; // it's OK, let isCandidate as true
        }
        else{
            // it's not OK, jump out to check the next bin!
            isCandidate = false; // invalidate this bin
            break;
        }
    } // iVar

    return isCandidate;
}
