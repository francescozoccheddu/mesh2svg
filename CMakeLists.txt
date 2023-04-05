cmake_minimum_required (VERSION 3.14)

include (FetchContent)
set (FETCHCONTENT_QUIET FALSE)
set(ALLOW_DUPLICATE_CUSTOM_TARGETS TRUE)

project ("meshviz")

# cinolib

set (CINOLIB_HEADER_ONLY OFF)
set (CINOLIB_USES_OPENGL_GLFW_IMGUI ON)

FetchContent_Declare (
    cinolib
    GIT_REPOSITORY "https://github.com/francescozoccheddu/cinolib.git"
    GIT_TAG "59fc0dd19e616568b5cd82c27920b6c09d3d11b4"
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable (cinolib)

# CGAL

find_package(CGAL)

# svg

file (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/svg)
file (DOWNLOAD "https://raw.githubusercontent.com/vincentlaucsb/svg/master/src/svg.hpp" ${CMAKE_CURRENT_BINARY_DIR}/svg/svg.hpp)

# main

add_executable (main "main.cpp")

set_target_properties (main PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries (main PRIVATE 
    cinolib
    CGAL::CGAL
)

target_include_directories(main PUBLIC 
    ${CMAKE_CURRENT_BINARY_DIR}/svg
)