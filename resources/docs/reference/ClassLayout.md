C++ Object Design and Layout
============================

The layout of an C++ object in memory is implementation dependent, but
both gcc and clang use the same ABI.  Other compilers might use a
different layout, but this one is relatively general. There are two
considerations.

   * The number of bytes needed by the class.  This is the sum of the data
     in the class.

   * The alignment alignment of the class.  This is determined by the
     largest alignment of the fundamental typles in the object (imprecisely,
     char is 8 bit, short is 16 bit, int is 32 bit, and pointer is 64 bit).

Simple inheritance rules that are usually work for class design:

   * There is one vtable pointer per object (forces 64 bit alignment)
      * (Almost) Always use a virtual deconstructor (forces a vtable).
        A virtual deconstructor allows the right deconstructor to be
        determined at run-time, while a non-virtual forces it to be
        determined at compile time.
   * Don't create holes in the class layout: larger alignment at the
     beginning; and, smaller alignment at the end.
      * {char, short, char} --> 6 bytes
      * {shart, char, char} --> 4 bytes

A slightly longer "rule" for class design:
   * When possible, plan the base class to match the derived classes
      * Additional virtual methods in the base class are "free".
      * Make the deconstructor virtual since you've already paid the
        "price for inheritance".
      * Try to make downcasting unnecesssary.
      * Class trees where derived classes modify methods are much cleaner
        than ones where the derived classes add methods.
         * Designs where derived classes that make the base class behavior
           more specific are better than ones that generalize the base class.
         * Downcasting is messy, and unreliable.  If methods are needed, put
           them in the base.
         * If two derived classes need different methods, revisit the class
           conceptual design.  Keep derived class methods uniform (seldom
           add public methods that aren't available in the base).

## Class Layout Details

Note: In gcc, you can examine the class layout using -fdump-lang-class

Assuming a vtable exists (we need virtual deconstructors, so it will)
and ignoring multiple inheritance since it can adds size and
complexity (not much of a loss since it is [almost] never needed), the
layout of a class is straightward.  Assume two classes

class base {
  base() {}
  virtual ~base() {}
  virtual int getA() {return a;}
  virtual int getAA() {return a*a;}
  int a;
}

class derived {
  derived() {}
  virtual ~derived() {}
  int getA() override {return a;}
  virtual getB() {return b;}
  int b;
}

These are mapped as

   * offsets for base (size is 16)
     ** -8-0: vtable pointer
     **  0-3: a
     **  4-8: not used

   * offsets for derived (size is 16)
     ** -8-0: vtable pointer
     **  0-3: a
     **  4-8: b

The virtual tables (vtables). [Simplifying for brevity]

vtable for base
   * 0: deconstructor
   * 8: getA
   * 16: getAA

vtable for derived
   * 0: deconstructor
   * 8: getA
   * 16: getAA
   * 24: getB

For multiple inheritance, the simplified "rule" is that each base
class adds another vtable pointer to the object.  The actual rule is
more complicated, but the object usually takes more space.
