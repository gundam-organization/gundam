set( LIB_NAME GundamDialDictionary )

set( SRCFILES
    src/DialBase.cpp
    src/DialBaseFactory.cpp
    src/NormDialBaseFactory.cpp
    src/GraphDialBaseFactory.cpp
    src/SplineDialBaseFactory.cpp
    src/DialInputBuffer.cpp
    src/DialInterface.cpp
    src/DialResponseSupervisor.cpp
    src/DialCollection.cpp
    src/EventDialCache.cpp
    src/Spline.cpp
    src/Graph.cpp
    src/LightGraph.cpp
    src/GeneralSpline.cpp
    src/UniformSpline.cpp
    src/CompactSpline.cpp
    src/MonotonicSpline.cpp
    src/SimpleSpline.cpp
    include/LightGraphCache.h)

set( HEADERS
    include/DialTypes.h
    include/DialBaseFactory.h
    include/NormDialBaseFactory.h
    include/GraphDialBaseFactory.h
    include/SplineDialBaseFactory.h
    include/DialBase.h
    include/DialInputBuffer.h
    include/DialInterface.h
    include/DialResponseSupervisor.h
    include/DialCollection.h
    include/DialUtils.h
    include/EventDialCache.h
    include/Norm.h
    include/Spline.h
    include/SplineCache.h
    include/Graph.h
    include/GraphCache.h
    include/LightGraph.h
    include/LightGraphCache.h
    include/GeneralSpline.h
    include/GeneralSplineCache.h
    include/UniformSpline.h
    include/UniformSplineCache.h
    include/CompactSpline.h
    include/CompactSplineCache.h
    include/MonotonicSpline.h
    include/MonotonicSplineCache.h
    include/SimpleSpline.h
    include/SimpleSplineCache.h
    )

if( USE_STATIC_LINKS )
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(
    ${LIB_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(
    ${LIB_NAME} PRIVATE
    ${ZLIB_INCLUDE_DIRS}
)

target_link_libraries(
    ${LIB_NAME}
    GundamFitParameters
    GundamFitSamples
    ${ZLIB_LIBRARIES}
    ${ROOT_LIBRARIES}
)



#set_target_properties(${LIB_NAME} PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS ${LIB_NAME} DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} GundamErrorPropagator PARENT_SCOPE)
