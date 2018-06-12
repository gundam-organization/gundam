#ifndef FITSTRUCTS_HH
#define FITSTRUCTS_HH

namespace xsllh
{

struct FitBin
{
    double D1low, D1high;
    double D2low, D2high;

    FitBin() : D1low(0), D1high(0), D2low(0), D2high(0) {}
    FitBin(const double D1_L, const double D1_H,
           const double D2_L, const double D2_H)
          : D1low(D1_L), D1high(D1_H),
            D2low(D2_L), D2high(D2_H)
          {}
};

};

#endif
