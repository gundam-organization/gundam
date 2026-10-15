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

  struct EvaluationBuffer{
    std::vector<int> variableIndices;
    std::vector<double> values;
    std::vector<BinFraction> fractions;
  };

  [[nodiscard]] bool isEnabled() const{ return _isEnabled_; }
  void build(const std::vector<Bin>& binList_);
  void prepareEvaluation(const VariableCollection& variables_, EvaluationBuffer& buffer_) const;
  void evaluate(const VariableCollection& variables_, EvaluationBuffer& buffer_) const;

protected:
  void configureImpl() override;

private:
  struct Axis{
    std::string name;
    double width{0};
  };
  struct ResponseInterval{
    size_t variableIndex{0};
    double lowerBound{0};
    double upperBound{0};
    double inverseWidth{0};
  };
  struct SelectionInterval{
    size_t variableIndex{0};
    double lowerBound{0};
    double upperBound{0};
  };
  struct ResponseBin{
    int binIndex{-1};
    std::vector<SelectionInterval> selectionIntervals;
    std::vector<ResponseInterval> intervals;
  };

  static double cumulativeKernel(double position_);
  [[nodiscard]] bool haveSameUnsmoothedEdges(const Bin& first_, const Bin& second_) const;
  [[nodiscard]] double findWidth(const std::string& variableName_) const;

  bool _isEnabled_{false};
  std::vector<Axis> _axes_;
  std::vector<ResponseBin> _responseBins_;
  std::vector<std::string> _variableNames_;
};

#endif
