# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ryz2/DanielWorkspace/test_libtorch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ryz2/DanielWorkspace/test_libtorch/build

# Include any dependencies generated for this target.
include CMakeFiles/test_torch.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_torch.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_torch.dir/flags.make

CMakeFiles/test_torch.dir/main.cpp.o: CMakeFiles/test_torch.dir/flags.make
CMakeFiles/test_torch.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ryz2/DanielWorkspace/test_libtorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_torch.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_torch.dir/main.cpp.o -c /home/ryz2/DanielWorkspace/test_libtorch/main.cpp

CMakeFiles/test_torch.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_torch.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ryz2/DanielWorkspace/test_libtorch/main.cpp > CMakeFiles/test_torch.dir/main.cpp.i

CMakeFiles/test_torch.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_torch.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ryz2/DanielWorkspace/test_libtorch/main.cpp -o CMakeFiles/test_torch.dir/main.cpp.s

# Object files for target test_torch
test_torch_OBJECTS = \
"CMakeFiles/test_torch.dir/main.cpp.o"

# External object files for target test_torch
test_torch_EXTERNAL_OBJECTS =

test_torch: CMakeFiles/test_torch.dir/main.cpp.o
test_torch: CMakeFiles/test_torch.dir/build.make
test_torch: /home/ryz2/DanielWorkspace/libtorch/lib/libtorch.so
test_torch: /home/ryz2/DanielWorkspace/libtorch/lib/libc10.so
test_torch: /home/ryz2/DanielWorkspace/libtorch/lib/libkineto.a
test_torch: /home/ryz2/DanielWorkspace/libtorch/lib/libc10.so
test_torch: CMakeFiles/test_torch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ryz2/DanielWorkspace/test_libtorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_torch"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_torch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_torch.dir/build: test_torch

.PHONY : CMakeFiles/test_torch.dir/build

CMakeFiles/test_torch.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_torch.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_torch.dir/clean

CMakeFiles/test_torch.dir/depend:
	cd /home/ryz2/DanielWorkspace/test_libtorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ryz2/DanielWorkspace/test_libtorch /home/ryz2/DanielWorkspace/test_libtorch /home/ryz2/DanielWorkspace/test_libtorch/build /home/ryz2/DanielWorkspace/test_libtorch/build /home/ryz2/DanielWorkspace/test_libtorch/build/CMakeFiles/test_torch.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_torch.dir/depend

