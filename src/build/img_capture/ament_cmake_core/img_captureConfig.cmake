# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_img_capture_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED img_capture_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(img_capture_FOUND FALSE)
  elseif(NOT img_capture_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(img_capture_FOUND FALSE)
  endif()
  return()
endif()
set(_img_capture_CONFIG_INCLUDED TRUE)

# output package information
if(NOT img_capture_FIND_QUIETLY)
  message(STATUS "Found img_capture: 0.0.1 (${img_capture_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'img_capture' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT img_capture_DEPRECATED_QUIET)
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(img_capture_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${img_capture_DIR}/${_extra}")
endforeach()
