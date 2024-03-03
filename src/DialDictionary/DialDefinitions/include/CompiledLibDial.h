//
// Created by Nadrino on 03/03/2024.
//

#ifndef GUNDAM_COMPILED_LIB_DIAL_H
#define GUNDAM_COMPILED_LIB_DIAL_H

#include "DialBase.h"

class CompiledLibDial : public DialBase {

public:
  CompiledLibDial() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<CompiledLibDial>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"RootFormula"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;
  [[nodiscard]] std::string getSummary() const override;

  bool loadLibrary(const std::string& path_);

private:
  void* _loadedLibrary_{nullptr};
  void* _evalFct_{nullptr};

};


#endif //GUNDAM_COMPILED_LIB_DIAL_H
