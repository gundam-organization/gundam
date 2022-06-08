#ifndef BINMANAGER_H
#define BINMANAGER_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class BinManager
{
    public:
        BinManager();
        BinManager(std::string  filename, bool UseNutypeBeammode=false);
        BinManager(const BinManager& source_);

        int GetNbins() const;
        int SetBinning(const std::string& filename, bool UseNutypeBeammode=false);
        int GetBinIndex(const std::vector<double>& val) const;
        int GetBinIndex(const std::vector<double>& val, const int val_nutype, const int val_beammode) const;
        double GetBinWidth(const int i) const;
        double GetBinWidth(const int i, const int d) const;
        unsigned int GetDimension() const;
        std::vector<double> GetBinVector(const double d) const;
        std::vector<std::vector<int>>& GetBinNuTypeList() { return bin_nutype; };
        std::vector<std::vector<int>>& GetBinBeamModeList() { return bin_beammode; };
        std::vector<std::vector<std::pair<double, double>>> GetEdgeVector() const { return binList; };
        std::vector<std::pair<double, double>> GetEdgeVector(const int d) const { return binList.at(d); };
        void Print() const;
        void MergeBins(unsigned int groupSize_, int dimIndexToRebin_ = -1);

    private:
        bool CheckBinIndex(const int i, const int d, const double val) const;
        bool CheckBinIndex(const int i, const int d, const double val, const int val_nutype, const int val_beammode) const;

        unsigned int nbins;
        unsigned int dimension;
        std::string fname_binning;
        std::vector<std::vector<std::pair<double, double>>> binList;
        std::vector<std::vector<int>> bin_nutype;
        std::vector<std::vector<int>> bin_beammode;
};

#endif
