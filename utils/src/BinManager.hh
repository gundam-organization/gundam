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
        BinManager(const std::string& filename, bool UseNutypeBeammode=false);

        int GetNbins() const;
        int SetBinning(const std::string& filename, bool UseNutypeBeammode=false);
        int GetBinIndex(const std::vector<double>& val) const;
        int GetBinIndex(const std::vector<double>& val, const int val_nutype, const int val_beammode) const;
        double GetBinWidth(const int i) const;
        double GetBinWidth(const int i, const int d) const;
        std::vector<double> GetBinVector(const double d) const;
        std::vector<std::vector<std::pair<double, double>>> GetEdgeVector() const { return bin_edges; };
        std::vector<std::pair<double, double>> GetEdgeVector(const int d) const { return bin_edges.at(d); };
        void Print() const;

    private:
        bool CheckBinIndex(const int i, const int d, const double val) const;
        bool CheckBinIndex(const int i, const int d, const double val, const int val_nutype, const int val_beammode) const;

        unsigned int nbins;
        unsigned int dimension;
        std::string fname_binning;
        std::vector<std::vector<std::pair<double, double>>> bin_edges;
        std::vector<std::vector<int>> bin_nutype;
        std::vector<std::vector<int>> bin_beammode;
};

#endif
