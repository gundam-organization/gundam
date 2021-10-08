//
// Created by Nadrino on 12/04/2021.
//

#ifndef XSLLHFITTER_GENERALIZEDFITBIN_H
#define XSLLHFITTER_GENERALIZEDFITBIN_H

#include <vector>

class GeneralizedFitBin
{

public:
    GeneralizedFitBin();
    GeneralizedFitBin(const std::vector<double>& lowBinEdgeList_, const std::vector<double>& highBinEdgeList_);

    void setBinEdges(const std::vector<double>& lowBinEdgeList_, const std::vector<double>& highBinEdgeList_);

    size_t getNbDimensions() const;
    double getBinLowEdge(size_t dimensionIndex_) const;
    double getBinHighEdge(size_t dimensionIndex_) const;
    const std::vector<double>& getLowBinEdgeList() const;
    const std::vector<double>& getHighBinEdgeList() const;

    bool isInBin(const std::vector<double>& eventVarList_) const;

private:
    std::vector<double> _lowBinEdgeList_{};
    std::vector<double> _highBinEdgeList_{};

};

#endif // XSLLHFITTER_GENERALIZEDFITBIN_H
