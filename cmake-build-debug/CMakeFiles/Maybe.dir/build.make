# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/alex/Desktop/CLION_WORKSPACE/Dijkstra_Parallel

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/alex/Desktop/CLION_WORKSPACE/Dijkstra_Parallel/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Maybe.dir/depend.make
# Include the progress variables for this target.
include CMakeFiles/Maybe.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Maybe.dir/flags.make

CMakeFiles/Maybe.dir/dijkstraMPI.cpp.o: CMakeFiles/Maybe.dir/flags.make
CMakeFiles/Maybe.dir/dijkstraMPI.cpp.o: ../dijkstraMPI.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/alex/Desktop/CLION_WORKSPACE/Dijkstra_Parallel/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Maybe.dir/dijkstraMPI.cpp.o"
	mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Maybe.dir/dijkstraMPI.cpp.o -c /Users/alex/Desktop/CLION_WORKSPACE/Dijkstra_Parallel/dijkstraMPI.cpp

CMakeFiles/Maybe.dir/dijkstraMPI.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Maybe.dir/dijkstraMPI.cpp.i"
	mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/alex/Desktop/CLION_WORKSPACE/Dijkstra_Parallel/dijkstraMPI.cpp > CMakeFiles/Maybe.dir/dijkstraMPI.cpp.i

CMakeFiles/Maybe.dir/dijkstraMPI.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Maybe.dir/dijkstraMPI.cpp.s"
	mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/alex/Desktop/CLION_WORKSPACE/Dijkstra_Parallel/dijkstraMPI.cpp -o CMakeFiles/Maybe.dir/dijkstraMPI.cpp.s

# Object files for target Maybe
Maybe_OBJECTS = \
"CMakeFiles/Maybe.dir/dijkstraMPI.cpp.o"

# External object files for target Maybe
Maybe_EXTERNAL_OBJECTS =

Maybe: CMakeFiles/Maybe.dir/dijkstraMPI.cpp.o
Maybe: CMakeFiles/Maybe.dir/build.make
Maybe: CMakeFiles/Maybe.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/alex/Desktop/CLION_WORKSPACE/Dijkstra_Parallel/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Maybe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Maybe.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Maybe.dir/build: Maybe
.PHONY : CMakeFiles/Maybe.dir/build

CMakeFiles/Maybe.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Maybe.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Maybe.dir/clean

CMakeFiles/Maybe.dir/depend:
	cd /Users/alex/Desktop/CLION_WORKSPACE/Dijkstra_Parallel/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/alex/Desktop/CLION_WORKSPACE/Dijkstra_Parallel /Users/alex/Desktop/CLION_WORKSPACE/Dijkstra_Parallel /Users/alex/Desktop/CLION_WORKSPACE/Dijkstra_Parallel/cmake-build-debug /Users/alex/Desktop/CLION_WORKSPACE/Dijkstra_Parallel/cmake-build-debug /Users/alex/Desktop/CLION_WORKSPACE/Dijkstra_Parallel/cmake-build-debug/CMakeFiles/Maybe.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Maybe.dir/depend

