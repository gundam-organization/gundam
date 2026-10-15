#ifndef GUNDAM_BINNING_RESPONSE_H
#define GUNDAM_BINNING_RESPONSE_H

#include "VariableCollection.h"
#include "Bin.h"

#include <string>
#include <vector>

class BinningResponse : public JsonBaseClass {
public:
  struct BinFraction{
    int binIndex{-1};
    double fraction{0};
  };

  [[nodiscard]] bool isEnabled() const{ return _isEnabled_; }
  void build(const std::vector<Bin>& binList_);
  void evaluate(const VariableCollection& variables_, std::vector<BinFraction>& fractions_) const;

protected:
  void configureImpl() override;

private:
  struct Axis{
    std::string name;
    double width{0};
  };
  struct ResponseInterval{
    std::string variableName;
    double lowerBound{0};
    double upperBound{0};
    double width{0};
  };
  struct ResponseBin{
    Bin bin;
    std::vector<ResponseInterval> intervals;
  };

  static double cumulativeKernel(double position_);
  [[nodiscard]] bool haveSameUnsmoothedEdges(const Bin& first_, const Bin& second_) const;
  [[nodiscard]] double findWidth(const std::string& variableName_) const;

  bool _isEnabled_{false};
  std::vector<Axis> _axes_;
  std::vector<ResponseBin> _responseBins_;
};

#endif
