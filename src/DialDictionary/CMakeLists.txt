set( LIB_NAME GundamDialDictionary )



set( SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/DialEngine/src/DialInputBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DialEngine/src/DialInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DialEngine/src/DialResponseSupervisor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DialEngine/src/DialCollection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DialEngine/src/EventDialCache.cpp

    # DialDefinitions
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/src/DialBase.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/src/Graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/src/LightGraph.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/src/Spline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/src/SimpleSpline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/src/GeneralSpline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/src/UniformSpline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/src/CompactSpline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/src/MonotonicSpline.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/src/RootFormula.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/src/Polynomial.cpp

    # DialFactories
    ${CMAKE_CURRENT_SOURCE_DIR}/DialFactories/src/DialBaseFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DialFactories/src/NormDialBaseFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DialFactories/src/GraphDialBaseFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DialFactories/src/SplineDialBaseFactory.cpp
    )

set( HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/DialEngine/include/DialInputBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialEngine/include/DialInterface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialEngine/include/DialResponseSupervisor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialEngine/include/DialCollection.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialEngine/include/EventDialCache.h

    # DialDefinitions
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include/DialBase.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include/DialTypes.h

    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include/Norm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include/Shift.h

    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include/Graph.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include/LightGraph.h

    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include/Spline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include/SimpleSpline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include/GeneralSpline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include/UniformSpline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include/CompactSpline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include/MonotonicSpline.h

    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include/RootFormula.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include/Polynomial.h

    # DialFactories
    ${CMAKE_CURRENT_SOURCE_DIR}/DialFactories/include/DialBaseFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialFactories/include/NormDialBaseFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialFactories/include/GraphDialBaseFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DialFactories/include/SplineDialBaseFactory.h
    )

if( USE_STATIC_LINKS )
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(
    ${LIB_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/DialEngine/include
    ${CMAKE_CURRENT_SOURCE_DIR}/DialFactories/include
    ${CMAKE_CURRENT_SOURCE_DIR}/DialDefinitions/include
)

#target_include_directories(
#    ${LIB_NAME} PRIVATE
#    ${ZLIB_INCLUDE_DIRS}
#)

target_link_libraries( ${LIB_NAME} PUBLIC
    GundamParametersManager
    GundamSamplesManager
)



#set_target_properties(${LIB_NAME} PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install( TARGETS ${LIB_NAME} DESTINATION lib )
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)
