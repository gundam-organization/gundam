#ifndef ConfigurationValue_h_Seen
#define ConfigurationValue_h_Seen
#include <ostream>

/// A variable that will remember a base value, and can be restored to the
/// base value.  Other than tracking the original value, ConfigurationValue
/// objects will behave like the fundamental type.  This is intended to save
/// configuration values that may be changed, and then reverted to the
/// original value.  The base value can be explicity set using the `set()`
/// method, or will take a value from the first time the `modify()` or
/// `operator =()` methods are used.
template <typename T>
class ConfigurationValue {
public:
    ConfigurationValue() {fSet = false;}
    ConfigurationValue(const T& v) {fSet = true; fCurrent = fBase = v;}
    operator T() const {return fCurrent;}
    operator T&() {return fCurrent;}

    /// Restore the current value to the base value and return the previous
    /// current value.  Do not use `(v.restore() == v)` since it can return
    /// either `true` or `false`.  It causes both `operator T()` and
    /// `restore()` to be called, but the order of operation is not defined by
    /// all C++ standards.
    T restore() {T old = fCurrent; fCurrent = fBase; return old;}

    /// Assign the current value.  If the base value has not been set,
    /// then this also assigns the base value.
    ConfigurationValue& operator =(const T& val) {
        modify(val);
        return *this;
    }

    /// Modify the current value without changing the base value.  If the base
    /// value has not been set, then also set the base value.  This returns
    /// the previous value.  The `operator =()` is the preferred way to modify
    /// the current value.  This is equivalent to the `operator =()`, except
    /// that it returns the old value instead of the current value.
    T modify(const T& val) {
        T old = fCurrent;
        if (not fSet) set(val);
        fCurrent = val;
        return old;
    }

    /// Set the base and current value.  This is the preferred way to set the
    /// base value.
    T set(const T& val) {T o=fBase; fBase=fCurrent=val; fSet=true; return o;}

    /// Get the current value.
    T get() const {return fCurrent;}

    /// Get the base value that will be used by the restore method.
    T getBase() const {return fBase;}

    /// Set the base value without modifying the current value.  This is
    /// provided for specific special cases and should be avoided.  User code
    /// should prefer `set()`.
    T setBase(T v) {T old = fBase; fBase = v; fSet = true; return old;}

private:
    T fBase{};
    T fCurrent{};
    bool fSet{false};
};

template<typename T>
std::ostream& operator<<(std::ostream& os, ConfigurationValue<T> v) {
    if (v.get() != v.getBase()) {
        os << "[Value: " << v.get() << ", Base: " << v.getBase() << "]";
    }
    else os << v.get();
    return os;
}
#endif
//  A Lesser GNU Public License

//  Copyright (C) 2023 Clark McGrew

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA
