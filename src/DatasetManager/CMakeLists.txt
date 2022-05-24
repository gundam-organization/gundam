set(SRCFILES
        src/DataSetLoader.cpp
        src/EventTreeWriter.cpp
        src/DataDispenser.cpp
)

set(HEADERS
        include/DatasetLoader.h
        include/EventTreeWriter.h
        include/DataDispenser.h
)

if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(GundamDatasetManager STATIC ${SRCFILES})
else()
  add_library(GundamDatasetManager SHARED ${SRCFILES})
endif()

target_include_directories(GundamDatasetManager PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries( GundamDatasetManager
        GundamFitParameters
        GundamFitSamples
        ${ROOT_LIBRARIES}
        )

set_target_properties(GundamDatasetManager PROPERTIES VERSION
  "${gundam_VERSION_MAJOR}.${gundam_VERSION_MINOR}.${gundam_VERSION_REVISION}${gundam_VERSION_TAG}")

install(TARGETS GundamDatasetManager DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

