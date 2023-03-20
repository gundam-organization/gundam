set( LIB_NAME GundamDialDictionary )

set( SRCFILES
    src/DialBase.cpp
    src/DialBaseCache.cpp
    src/DialInputBuffer.cpp
    src/DialInterface.cpp
    src/DialResponseSupervisor.cpp
    src/DialCollection.cpp
    src/SplineHandler.cpp
    src/EventDialCache.cpp
    src/GraphHandler.cpp
    src/LightGraphHandler.cpp
    src/GeneralSplineHandler.cpp
    src/UniformSplineHandler.cpp
    src/CompactSplineHandler.cpp
    src/MonotonicSplineHandler.cpp
    src/SimpleSplineHandler.cpp
    include/LightGraphCache.h)

set( HEADERS
    include/DialTypes.h
    include/DialBase.h
    include/DialBaseCache.h
    include/DialInputBuffer.h
    include/DialInterface.h
    include/DialResponseSupervisor.h
    include/DialCollection.h
    include/SplineHandler.h
    include/Norm.h
    include/Spline.h
    include/SplineCache.h
    include/DialUtils.h
    include/EventDialCache.h
    include/GraphHandler.h
    include/LightGraphHandler.h
    include/Graph.h
    include/LightGraph.h
    include/GeneralSplineHandler.h
    include/GeneralSpline.h
    include/GeneralSplineCache.h
    include/UniformSplineHandler.h
    include/UniformSpline.h
    include/CompactSplineHandler.h
    include/CompactSpline.h
    include/CompactSplineCache.h
    include/MonotonicSplineHandler.h
    include/MonotonicSpline.h
    include/MonotonicSplineCache.h
    include/SimpleSplineHandler.h
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
