cmake_minimum_required(VERSION 3.10.2)

project(NIFTI 
        VERSION 0.0.1
        LANGUAGES C)

# Basic options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Find required ZLIB
find_package(ZLIB REQUIRED)
add_definitions(-DHAVE_ZLIB)

# Find math library on Unix
if(NOT WIN32)
    find_library(MATH_LIB m)
endif()

# Add the two required subdirectories
add_subdirectory(znzlib)
add_subdirectory(nifti2)