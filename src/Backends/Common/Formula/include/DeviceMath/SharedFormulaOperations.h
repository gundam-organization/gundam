// Intentionally unguarded: expanded separately for unary and binary dispatch.
// Callers supply case generation; math and memory operations remain backend-specific.
#ifdef GUNDAM_FORMULA_UNARY
GUNDAM_FORMULA_UNARY(Negate, negate)
GUNDAM_FORMULA_UNARY(Exp, exp)
GUNDAM_FORMULA_UNARY(Log, log)
GUNDAM_FORMULA_UNARY(Log10, log10)
GUNDAM_FORMULA_UNARY(Sqrt, sqrt)
GUNDAM_FORMULA_UNARY(Abs, abs)
GUNDAM_FORMULA_UNARY(Sin, sin)
GUNDAM_FORMULA_UNARY(Cos, cos)
GUNDAM_FORMULA_UNARY(Tan, tan)
GUNDAM_FORMULA_UNARY(Asin, asin)
GUNDAM_FORMULA_UNARY(Acos, acos)
GUNDAM_FORMULA_UNARY(Atan, atan)
GUNDAM_FORMULA_UNARY(Sinh, sinh)
GUNDAM_FORMULA_UNARY(Cosh, cosh)
GUNDAM_FORMULA_UNARY(Tanh, tanh)
GUNDAM_FORMULA_UNARY(Erf, erf)
GUNDAM_FORMULA_UNARY(Erfc, erfc)
GUNDAM_FORMULA_UNARY(Floor, floor)
GUNDAM_FORMULA_UNARY(Ceil, ceil)
#endif
#ifdef GUNDAM_FORMULA_BINARY
GUNDAM_FORMULA_BINARY(Add, add)
GUNDAM_FORMULA_BINARY(Subtract, subtract)
GUNDAM_FORMULA_BINARY(Multiply, multiply)
GUNDAM_FORMULA_BINARY(Divide, divide)
GUNDAM_FORMULA_BINARY(Pow, pow)
GUNDAM_FORMULA_BINARY(Atan2, atan2)
GUNDAM_FORMULA_BINARY(Min, min)
GUNDAM_FORMULA_BINARY(Max, max)
#endif
