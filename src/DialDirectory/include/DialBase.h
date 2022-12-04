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

  // to override
  [[nodiscard]] virtual std::unique_ptr<DialBase> clone() const = 0; // can't allocate pure virtual class
  [[nodiscard]] virtual std::string getDialTypeName() const { return {"DialBase"}; }
  virtual double evalResponseImpl(const DialInputBuffer& input_) = 0;

  // other virtual
  virtual double evalResponse(const DialInputBuffer& input_){ return this->evalResponseImpl(input_); }

  //! class size = 8 bytes (no padding!)
};






#endif //GUNDAM_DIALBASE_H
