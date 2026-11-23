# src/DialDictionary/CMakeLists.txt
#
# Builds the GundamDialDictionary library: the dials (response functions of
# the fit parameters), the collections that group them, the event-to-dial
# cache, the cache of dials that change event variables, and the dial
# factories (tables, kriging, external weights).
#
# Included from cmake/modules.cmake, which adds every src/ module in turn.
# Link order (see the library-layers diagram in the docs):
#   GundamPropagator -> GundamDialDictionary -> GundamSamplesManager
#                    -> GundamParametersManager -> GundamUtils

# Name of the library target built here.
set( LIB_NAME GundamDialDictionary )

# Source files compiled into the library.
# DialEngine/src/DialBuilder.cpp is deliberately absent: DialBuilder is an
# empty placeholder class that nothing uses.
set( SRCFILES
    DialEngine/src/DialManager.cpp
    DialEngine/src/DialInputBuffer.cpp
    DialEngine/src/DialInterface.cpp
    DialEngine/src/DialResponseSupervisor.cpp
    DialEngine/src/DialCollection.cpp
    DialEngine/src/EventDialCache.cpp
    DialEngine/src/EventVariableResponseCache.cpp

    # DialDefinitions
    DialDefinitions/src/DialUtils.cpp
    DialDefinitions/src/DialBase.cpp
    DialDefinitions/src/CompositeDial.cpp
    DialDefinitions/src/EventFormula.cpp

    DialDefinitions/src/Kriged.cpp

    DialDefinitions/src/RootGraph.cpp
    DialDefinitions/src/Graph.cpp

    DialDefinitions/src/RootSpline.cpp
    DialDefinitions/src/GeneralSpline.cpp
    DialDefinitions/src/UniformSpline.cpp
    DialDefinitions/src/CompactSpline.cpp
    DialDefinitions/src/MonotonicSpline.cpp
    DialDefinitions/src/Bilinear.cpp
    DialDefinitions/src/Bicubic.cpp

    DialDefinitions/src/CompiledLibDial.cpp
    DialDefinitions/src/ExternalWeightDispatcher.cpp
    DialDefinitions/src/RootFormula.cpp
    DialDefinitions/src/Polynomial.cpp

    # DialFactories
    DialFactories/src/TabulatedDialFactory.cpp
    DialFactories/src/KrigedDialFactory.cpp
    DialFactories/src/ExternalWeightDialFactory.cpp
    )

# Public headers of the library.
# Only used by the commented-out install(FILES ...) at the bottom; the headers
# are found through target_include_directories() below, not through this list.
# Not complete: DialManager.h, DialUtils.h, CompiledLibDial.h,
# DialFactoryBase.h and DialBuilder.h are missing.
set( HEADERS
    DialEngine/include/DialInputBuffer.h
    DialEngine/include/DialInterface.h
    DialEngine/include/DialResponseSupervisor.h
    DialEngine/include/DialCollection.h
    DialEngine/include/EventDialCache.h
    DialEngine/include/EventVariableResponseCache.h

    # DialDefinitions
    DialDefinitions/include/DialBase.h
    DialDefinitions/include/CompositeDial.h
    DialDefinitions/include/EventFormula.h

    DialDefinitions/include/Norm.h
    DialDefinitions/include/Shift.h
    DialDefinitions/include/Tabulated.h
    DialDefinitions/include/Kriged.h

    DialDefinitions/include/RootGraph.h
    DialDefinitions/include/Graph.h

    DialDefinitions/include/RootSpline.h
    DialDefinitions/include/GeneralSpline.h
    DialDefinitions/include/UniformSpline.h
    DialDefinitions/include/CompactSpline.h
    DialDefinitions/include/MonotonicSpline.h
    DialDefinitions/include/Bilinear.h
    DialDefinitions/include/Bicubic.h

    DialDefinitions/include/RootFormula.h
    DialDefinitions/include/Polynomial.h
    DialDefinitions/include/ExternalWeightDispatcher.h

    # DialFactories
    DialFactories/include/TabulatedDialFactory.h
    DialFactories/include/KrigedDialFactory.h
    DialFactories/include/ExternalWeightDialFactory.h
    )

# Static or shared library, chosen by the USE_STATIC_LINKS option
# (cmake/options.cmake, default OFF = shared).
if( USE_STATIC_LINKS )
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

# Header folders, exported to every target that links this library
# (for example GundamPropagator and the CacheManager). BEFORE puts them in
# front of the include paths the target already has.
target_include_directories(
    ${LIB_NAME}
    BEFORE
    PUBLIC
    DialEngine/include
    DialFactories/include
    DialDefinitions/include
)

# Dials need the parameters (GundamParametersManager) and the events and
# samples (GundamSamplesManager), which in turn bring the utilities and ROOT.
target_link_libraries( ${LIB_NAME}
    PUBLIC
    GundamParametersManager
    GundamSamplesManager
)

# shm_open/shm_unlink are provided by librt on Linux. They are part of libc on
# macOS, so do not add this dependency there.
if( UNIX AND NOT APPLE )
  target_link_libraries( ${LIB_NAME} PRIVATE rt )
endif()



#set_target_properties(${LIB_NAME} PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

# Install the compiled library into <install prefix>/lib.
install( TARGETS ${LIB_NAME} DESTINATION lib )
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

# Appends to the MODULETargets list in the parent scope.
# Note: no CMake file in the repository reads MODULETargets, and the name
# appended (GundamErrorPropagator) is not this library's name; the line looks
# like a leftover and has no effect on the build.
set(MODULETargets ${MODULETargets} GundamErrorPropagator PARENT_SCOPE)
