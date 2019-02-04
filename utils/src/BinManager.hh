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
        BinManager(const std::string& filename);

        int GetNbins() const;
        int SetBinning(const std::string& filename);
        int GetBinIndex(const std::vector<double>& val) const;
        double GetBinWidth(const int i) const;
        double GetBinWidth(const int i, const int d) const;
        std::vector<double> GetBinVector(const double d) const;
        std::vector<std::vector<std::pair<double, double>>> GetEdgeVector() const { return bin_edges; };
        void Print() const;

    private:
        bool CheckBinIndex(const int i, const int d, const double val) const;

        unsigned int nbins;
        unsigned int dimension;
        std::string fname_binning;
        std::vector<std::vector<std::pair<double, double>>> bin_edges;
};

#endif
