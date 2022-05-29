set( LIB_NAME GundamDialDirectory )

set( SRCFILES
        src/DialDirectory.cpp
        src/DialCollection.cpp
        src/DialBase.cpp
        src/NormDial.cpp
        src/NestedDial.cpp
)

set( HEADERS
        include/DialDirectory.h
        include/DialCollection.h
        include/DialBase.h
        include/NormDial.h
        include/NestedDial.h
)

if( USE_STATIC_LINKS )
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(${LIB_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries( ${LIB_NAME} GundamFitParameters ${ROOT_LIBRARIES} )



#set_target_properties(${LIB_NAME} PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS ${LIB_NAME} DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} GundamErrorPropagator PARENT_SCOPE)
