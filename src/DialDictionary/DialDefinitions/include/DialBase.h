//
// Created by Adrien BLANCHET on 13/04/2022.
//

#ifndef GUNDAM_DIALBASE_H
#define GUNDAM_DIALBASE_H

#include "DialInputBuffer.h"

#include <vector>
#include <string>
#include <memory>

// should be thread safe -> add lock?
// any number of inputs (provided doubles) -> set input size
// fast -> no checks while eval

// as light as possible: minimize members
// keep in mind: class size is n*8 bytes (64 bits chunks)
// virtual layer adds 8 bytes for vftable pointer (virtual function table)
// adds members memory ->
// https://stackoverflow.com/questions/9439240/sizeof-class-with-int-function-virtual-function-in-c

class DialBase {
public:
  DialBase() = default;
  virtual ~DialBase() = default;

  // virtual layer + 8 bytes

  // Construct a copy of this dial.  Needed for PolymorphicObject.
  [[nodiscard]] virtual std::unique_ptr<DialBase> clone() const = 0;

  // Return the name of the dial type (simple local RTTI).
  [[nodiscard]] virtual std::string getDialTypeName() const = 0;

  /// Evaluate the dial response at the set of parameter values in DialInputBuffer.
  [[nodiscard]] virtual double evalResponse(const DialInputBuffer& input_) const = 0;

  /// Allow extrapolation of the data. The default should be to forbid extrapolation.
  virtual void setAllowExtrapolation(bool allow_){}
  [[nodiscard]] virtual bool getAllowExtrapolation() const {return false;}

  /// Dial summary describing its content
  [[nodiscard]] virtual std::string getSummary() const { return {}; };

  /////////////////////////////////////////////////////////////////////////
  // Pass information to the dial so that it can build it's internal
  // information.  New overloads should be added as we have classes of dials
  // (e.g. multi-dimensional dials).
  /////////////////////////////////////////////////////////////////////////

  /// Return the data used by the dial to calculate the output values. The
  /// specific data contained in the vector depends on the derived class.
  [[nodiscard]] virtual const std::vector<double>& getDialData() const;

};

#endif //GUNDAM_DIALBASE_H
// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
