set( LIB_NAME GundamDials )

set( SRCFILES
    src/DialBase.cpp
    src/DialBaseCache.cpp
    src/DialInputBuffer.cpp
    src/DialInterface.cpp
    src/DialResponseSupervisor.cpp
    src/DialCollection.cpp
    src/DialSplineHandler.cpp
    src/Norm.cpp
    src/NormBinned.cpp
    src/SplineCacheBinned.cpp
    src/DialBinned.cpp
    src/Spline.cpp
    src/SplineCache.cpp
    )

set( HEADERS
    include/DialBase.h
    include/DialBaseCache.h
    include/DialInputBuffer.h
    include/DialInterface.h
    include/DialResponseSupervisor.h
    include/DialCollection.h
    include/DialSplineHandler.h
    include/Norm.h
    include/NormBinned.h
    include/SplineCacheBinned.h
    include/DialBinned.h
    include/Spline.h
    include/SplineCache.h
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

target_link_libraries(
    ${LIB_NAME}
    GundamFitParameters
    GundamFitSamples
    ${ROOT_LIBRARIES}
)



#set_target_properties(${LIB_NAME} PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS ${LIB_NAME} DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} GundamErrorPropagator PARENT_SCOPE)
