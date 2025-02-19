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

  /// Evaluate the the dial response at the set of parameter values in
  /// DialInputBuffer.
  [[nodiscard]] virtual double evalResponse(const DialInputBuffer& input_) const = 0;

  /// Allow extrapolation of the data.  The default is to
  /// forbid extrapolation.
  virtual void setAllowExtrapolation(bool allow_) {}
  [[nodiscard]] virtual bool getAllowExtrapolation() const {return false;}

  /// Dial summary describing its content
  [[nodiscard]] virtual std::string getSummary() const { return {}; };

  /////////////////////////////////////////////////////////////////////////
  // Pass information to the dial so that it can build it's internal
  // information.  New overloads should be added as we have classes of dials
  // (e.g. multi-dimensional dials).
  /////////////////////////////////////////////////////////////////////////

  /// Build the dial with no input arguments.  This is mostly for competeness!
  virtual void buildDial(const std::string& option_="") {throw std::runtime_error("Not implemented");}

  /// Build the dial using up to three vectors of doubles.  This is how to add
  /// an array of one, two or three values to a dial (if only one or two
  /// vectors are needed, then the classes buildDial should just ignore the
  /// extra vectors).
  virtual void buildDial(const std::vector<double>& v1,
                         const std::vector<double>& v2,
                         const std::vector<double>& v3,
                         const std::string& option_="") {throw std::runtime_error("Not implemented");}

  /// Build the dial using a graph (usually a leaf in the input file).
  virtual void buildDial(const TGraph& grf, const std::string& option_="") {throw std::runtime_error("Not implemented");}

  /// Build the dial using a TSpline3 (usually a leaf in the input file).
  virtual void buildDial(const TSpline3& spl, const std::string& option_="") {throw std::runtime_error("Not implemented");}

  /// Build the dial using a double.  This is used on a "constant" dial like
  /// Shift, but can also be used in a dial that might do something like
  /// calculate the oscillation probability where the value could be closing
  /// over the event L/E.
  virtual void buildDial(double v1, const std::string& option_="") {throw std::runtime_error("Not implemented");}

  /// Build a dial from a TH2 (usually a leaf in the input file).
  virtual void buildDial(const TH2& h2, const std::string& option_="") {throw std::runtime_error("Not implemented");}

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
