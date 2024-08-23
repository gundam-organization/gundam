API Changes
===========

API changes are a good way to introduce bugs since they end up changing assumptions that old code has made.  GUNDAM is mostly self-contained, but API changes can also cause knock-on in user's libraries.  If you need to change an API, please follow the best practices.

# Best Practices

You can find a lot of discussion about the best practices for deprecating a source interface (API) in software design literature, but they point down to:

1) Discuss and plan the new interface.  Reach agreement and consider unintended effects.  Be aware that it is seldom possible to identify every place the old interface is used.

  1) Never remove functionality without a replacement.

2) Implement and test the new interface. (at least two stable releases before deprecation)

4) (Often) Use the new interface to implement a source compatible version of the old interface.  (at least one or two stable releases before deprecation) 

3) Sunset interface and make it produce warnings if it is being used.  The sunset interface needs to continue working until removed.  Warnings can be produced using compiler directives (e.g. "warning", or attributes "[[deprecated]]".  (at least two stable release before deprecation)

4) Deprecate the old interface and flag use during run-time with a warning message.  For extra points, error messages should describe replacements (the deprecation release).  The deprecated interface should not be called by any run-time code.

5) Do not remove the deprecated interface for several releases (at least two stable releases).

And as always test, test, test.

# Important corollaries:

Think very carefully about your class interfaces when you design the class.  In particular, examine your assumptions, and think about how the class might be generalized.  Make sure you choose good names that *mean* what you intend them to mean, and which follow the conventions of the library or application you are working on.
