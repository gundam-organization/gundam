//
// Created by Adrien BLANCHET on 13/04/2022.
//

#ifndef GUNDAM_DIALBASE_H
#define GUNDAM_DIALBASE_H

#include "DialInputBuffer.h"

#include "vector"
#include "string"
#include "memory"

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

  /// Evaluate the the dial response at the set of parameter values in
  /// DialInputBuffer.
  [[nodiscard]] virtual double evalResponse(const DialInputBuffer& input_) const = 0;

  /// Allow extrapolation of the data.  The default is to
  /// forbid extrapolation.
  virtual void setAllowExtrapolation(bool allow=false) {}
  virtual bool getAllowExtrapolation() const {return false;}

  /////////////////////////////////////////////////////////////////////////
  // Pass information to the dial so that it can build it's internal
  // information.  New overloads should be added as we have classes of dials
  // (e.g. multi-dimensional dials).
  /////////////////////////////////////////////////////////////////////////

  /// Build the dial with no input arguments.  Mostly here for completeness!
  virtual void buildDial(std::string option="") {throw std::runtime_error("Not implemented");}

  /// Build the dial using up to up to three vectors of doubles.  Mostly
  /// an internal tool to make spline and graph builders work the same.
  virtual void buildDial(const std::vector<double>& v1,
                         const std::vector<double>& v2,
                         const std::vector<double>& v3,
                         std::string option="") {throw std::runtime_error("Not implemented");}

  /// Build the dial using a graph (usually a leaf in the input file
  virtual void buildDial(const TGraph& grf, std::string option="") {throw std::runtime_error("Not implemented");}

  /// Build the dial using a spline (usually a leaf in the input file.
  virtual void buildDial(const TSpline3& spl, std::string option="") {throw std::runtime_error("Not implemented");}

  /// Return the data used by the dial to calculate the output values. The
  /// specific data contained in the vector depends on the derived class.
  [[nodiscard]] virtual const std::vector<double>& getDialData() const;

};

// extensions
#include "CachedDial.h"

#endif //GUNDAM_DIALBASE_H
// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
