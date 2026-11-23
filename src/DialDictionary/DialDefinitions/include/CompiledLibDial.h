//
// Created by Nadrino on 03/03/2024.
//

/// @file CompiledLibDial.h
/// @brief Declares CompiledLibDial, a dial whose response comes from a
///        function in a user-supplied shared library.
/// @ingroup DialDictionary

#ifndef GUNDAM_COMPILED_LIB_DIAL_H
#define GUNDAM_COMPILED_LIB_DIAL_H

#include "DialBase.h"

/// Dial whose response is computed by a function in a user-supplied shared
/// library.
///
/// A dial is a response function of one or more fit parameters (see
/// DialBase). DialCollection::makeDial(const ConfigReader&, const Bin*)
/// creates one CompiledLibDial for a `dialType: CompiledLibDial` collection
/// and loads the library named by `options/libraryFile` (`dialConfig` is
/// accepted for `options`). The library must export a
/// function with C linkage named `evalVariable`, of type
/// `double evalVariable(double* inputs)`. It receives the dial inputs and
/// returns the response. The inputs are the parameters listed in the
/// collection's `dialInputList`, in order (without that option, the single
/// parameter the dial is attached to); the function is not told how many
/// there are.
///
/// Not supported by Cache::Manager (no GPU implementation).
///
/// @par Ownership
/// The library handle is never closed (there is no dlclose()). Copies made by
/// clone() share the handle and the function pointer.
///
/// @par Thread safety
/// evalResponse() holds the global GUNDAM mutex
/// (GundamGlobals::getGlobalMutEx()) during the library call, so the user
/// function never runs on two threads at once. As a consequence, threads that
/// evaluate these dials wait for each other.
/// @ingroup DialDictionary
class CompiledLibDial : public DialBase {

public:
  /// Creates a dial with no library.
  /// Call loadLibrary() before evaluating.
  CompiledLibDial() = default;

  /// Copies this dial.
  /// @return A new CompiledLibDial sharing the same library handle and
  ///         function pointer.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<CompiledLibDial>(*this); }

  /// Gives the type name of this dial.
  /// @return `"RootFormula"`.
  /// @warning This is the same name as RootFormula's, which looks like a
  ///          copy-paste slip: summaries and error messages label these dials
  ///          "RootFormula".
  [[nodiscard]] std::string getDialTypeName() const override { return {"RootFormula"}; }

  /// Calls the library's `evalVariable` function on the dial inputs.
  /// @param input_ Buffer holding the inputs; a pointer to its first value is
  ///        passed to the function.
  /// @return The value returned by `evalVariable`.
  /// @pre loadLibrary() returned true; otherwise the function pointer is null.
  /// @warning The pointer is passed as non-const (const is cast away), so a
  ///          library that writes through it would change the input buffer
  ///          shared with the other dials of the collection.
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  /// Opens a shared library and looks up its `evalVariable` symbol.
  /// @param path_ Path to the library file. This function does not expand
  ///        environment variables, and the path must name a regular file:
  ///        GenericToolbox::isFile() uses lstat(), so a symbolic link is
  ///        rejected.
  /// @return True on success. False, with an error logged, if the file is not
  ///         found, dlopen() fails or the symbol is missing; DialCollection
  ///         then throws (LogThrow()).
  bool loadLibrary(const std::string& path_);

private:
  /// Handle returned by dlopen(); never closed.
  void* _loadedLibrary_{nullptr};

  /// Address of the library's `evalVariable` function, cast to
  /// `double(*)(double*)` when called. Null until loadLibrary() succeeds.
  void* _evalFct_{nullptr};

};


#endif //GUNDAM_COMPILED_LIB_DIAL_H
