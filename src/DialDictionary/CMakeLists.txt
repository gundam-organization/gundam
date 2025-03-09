set( LIB_NAME GundamDialDictionary )

set( SRCFILES
    DialEngine/src/DialManager.cpp
    DialEngine/src/DialInputBuffer.cpp
    DialEngine/src/DialInterface.cpp
    DialEngine/src/DialResponseSupervisor.cpp
    DialEngine/src/DialCollection.cpp
    DialEngine/src/EventDialCache.cpp

    # DialDefinitions
    DialDefinitions/src/DialUtils.cpp
    DialDefinitions/src/DialBase.cpp

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
    DialDefinitions/src/RootFormula.cpp
    DialDefinitions/src/Polynomial.cpp

    # DialFactories
    DialFactories/src/TabulatedDialFactory.cpp
    )

set( HEADERS
    DialEngine/include/DialInputBuffer.h
    DialEngine/include/DialInterface.h
    DialEngine/include/DialResponseSupervisor.h
    DialEngine/include/DialCollection.h
    DialEngine/include/EventDialCache.h

    # DialDefinitions
    DialDefinitions/include/DialBase.h

    DialDefinitions/include/Norm.h
    DialDefinitions/include/Shift.h
    DialDefinitions/include/Tabulated.h

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

    # DialFactories
    DialFactories/include/DialBaseFactory.h
    DialFactories/include/NormDialBaseFactory.h
    DialFactories/include/GraphDialBaseFactory.h
    DialFactories/include/SplineDialBaseFactory.h
    DialFactories/include/SurfaceDialBaseFactory.h
    DialFactories/include/TabulatedDialFactory.h
    )

if( USE_STATIC_LINKS )
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(
    ${LIB_NAME} PUBLIC
    DialEngine/include
    DialFactories/include
    DialDefinitions/include
)

target_link_libraries( ${LIB_NAME}
    PUBLIC
    GundamSamplesManager
)



#set_target_properties(${LIB_NAME} PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install( TARGETS ${LIB_NAME} DESTINATION lib )
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} GundamErrorPropagator PARENT_SCOPE)
