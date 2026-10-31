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

  struct EventResponse{
    std::vector<BinFraction> binFractions;
    double outsideFraction{1};
  };

  struct EvaluationBuffer{
    std::vector<int> variableIndices;
    std::vector<double> values;
    EventResponse response;
  };

  [[nodiscard]] const auto& getVariableNames() const{ return _variableNames_; }
  [[nodiscard]] const auto& getResponseBins() const{ return _responseBins_; }
  [[nodiscard]] const auto& getRelativeResponseBins() const{ return _relativeResponseBins_; }
  [[nodiscard]] bool isEnabled() const{ return _isEnabled_; }
  [[nodiscard]] bool hasRelativeWidths() const{ return _hasRelativeWidths_; }
  void build(const std::vector<Bin>& binList_);
  void prepareEvaluation(const VariableCollection& variables_, EvaluationBuffer& buffer_) const;
  void evaluate(const VariableCollection& variables_, EvaluationBuffer& buffer_) const;
  void evaluateAbsolute(const VariableCollection& variables_, EvaluationBuffer& buffer_) const;
  void evaluateRelative(const VariableCollection& variables_, EvaluationBuffer& buffer_) const;

protected:
  void configureImpl() override;

private:
  struct Axis{
    std::string name;
    double width{0};
    double widthFraction{0};
  };
  struct ResponseInterval{
    size_t variableIndex{0};
    double lowerBound{0};
    double upperBound{0};
    double inverseWidth{0};
  };
  struct RelativeResponseInterval{
    size_t variableIndex{0};
    double lowerBound{0};
    double upperBound{0};
    double lowerInverseWidth{0};
    double upperInverseWidth{0};
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

  struct RelativeResponseBin{
    int binIndex{-1};
    std::vector<SelectionInterval> selectionIntervals;
    std::vector<RelativeResponseInterval> intervals;
  };

  void buildAbsoluteResponse(const std::vector<Bin>& binList_);
  void buildRelativeResponse(const std::vector<Bin>& binList_);
  [[nodiscard]] bool haveOverlappingSelections(const Bin& first_, const Bin& second_) const;
  [[nodiscard]] std::vector<SelectionInterval> buildSelectionIntervals(const Bin& bin_) const;
  [[nodiscard]] double evaluateRelativeResponse(EvaluationBuffer& buffer_) const;
  static double cumulativeKernel(double position_);
  [[nodiscard]] const Axis* findAxis(const std::string& variableName_) const;

  bool _isEnabled_{false};
  bool _hasRelativeWidths_{false};
  std::vector<Axis> _axes_;
  std::vector<ResponseBin> _responseBins_;
  std::vector<std::string> _variableNames_;
  std::vector<RelativeResponseBin> _relativeResponseBins_;
};

#endif
