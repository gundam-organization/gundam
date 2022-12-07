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
    )

set( HEADERS
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
    include/Graph.h
    include/GraphHandler.h
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
