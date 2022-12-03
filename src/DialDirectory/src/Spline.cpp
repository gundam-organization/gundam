//
// Created by Adrien Blanchet on 30/11/2022.
//

#include "Spline.h"

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[Spline]");
});

double Spline::evalResponseImpl(const DialInputBuffer& input_) {
//  LogTrace << GET_VAR_NAME_VALUE(input_.getBuffer()[0]) << " -> " << _spline_.Eval( input_.getBuffer()[0] ) << std::endl;
  return _spline_.Eval( input_.getBuffer()[0] );
}
