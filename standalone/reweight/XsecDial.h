#ifndef XSECDIAL_H
#define XSECDIAL_H

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <TCollection.h>
#include <TFile.h>
#include <TGraph.h>
#include <TKey.h>

#include "BinManager.h"

class XsecDial
{
    public:
        XsecDial(const std::string& dial_name);
        XsecDial(const std::string& dial_name, const std::string& fname_splines,
                 const std::string& fname_binning);

        void SetBinning(const std::string& fname_binning);
        void ReadSplines(const std::string& fname_splines);

        int GetSplineIndex(int topology, int reaction, double q2) const;
        int GetSplineIndex(const std::vector<int>& var, const std::vector<double>& bin) const;
        double GetSplineValue(int index, double dial_value) const;

        void SetVars(double nominal, double step, double limit_lo, double limit_hi);
        void SetDimensions(int num_top, int num_reac);
        void SetDimensions(const std::vector<int>& dim);
        void Print(bool print_bins = false) const;

        std::string GetName() const { return m_name; }
        double GetNominal() const { return m_nominal; }
        double GetStep() const { return m_step; }
        double GetLimitLow() const { return m_limit_lo; }
        double GetLimitHigh() const { return m_limit_hi; }

    private:
        int ntop;
        int nreac;
        int nbins;
        std::string m_name;
        double m_nominal;
        double m_step;
        double m_limit_lo;
        double m_limit_hi;
        std::vector<TGraph> v_splines;
        std::vector<int> m_dimensions;
        BinManager bin_manager;

        const std::string TAG;
};

#endif
