#ifndef BINMANAGER_H
#define BINMANAGER_H

#include <algorithm>
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
        std::vector<double> GetBinVector(const double d) const;
        void Print() const;

    private:
        bool CheckBinIndex(const int i, const int d, const double val) const;

        unsigned int nbins;
        unsigned int dimension;
        std::string fname_binning;
        std::vector<std::vector<std::pair<double, double>>> bin_edges;
};

#endif
