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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /lclhome/tcickovs/PluMA/pipelines/GPU

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /lclhome/tcickovs/PluMA/pipelines/GPU/build

# Include any dependencies generated for this target.
include CMakeFiles/Reduction_DatasetGenerator.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Reduction_DatasetGenerator.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Reduction_DatasetGenerator.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Reduction_DatasetGenerator.dir/flags.make

CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.o: CMakeFiles/Reduction_DatasetGenerator.dir/flags.make
CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.o: ../Module9/Reduction/dataset_generator.cpp
CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.o: CMakeFiles/Reduction_DatasetGenerator.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/lclhome/tcickovs/PluMA/pipelines/GPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.o -MF CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.o.d -o CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.o -c /lclhome/tcickovs/PluMA/pipelines/GPU/Module9/Reduction/dataset_generator.cpp

CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /lclhome/tcickovs/PluMA/pipelines/GPU/Module9/Reduction/dataset_generator.cpp > CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.i

CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /lclhome/tcickovs/PluMA/pipelines/GPU/Module9/Reduction/dataset_generator.cpp -o CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.s

# Object files for target Reduction_DatasetGenerator
Reduction_DatasetGenerator_OBJECTS = \
"CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.o"

# External object files for target Reduction_DatasetGenerator
Reduction_DatasetGenerator_EXTERNAL_OBJECTS =

CMakeFiles/Reduction_DatasetGenerator.dir/cmake_device_link.o: CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.o
CMakeFiles/Reduction_DatasetGenerator.dir/cmake_device_link.o: CMakeFiles/Reduction_DatasetGenerator.dir/build.make
CMakeFiles/Reduction_DatasetGenerator.dir/cmake_device_link.o: libgputk.a
CMakeFiles/Reduction_DatasetGenerator.dir/cmake_device_link.o: /usr/local/cuda-11.2/lib64/libcudart.so
CMakeFiles/Reduction_DatasetGenerator.dir/cmake_device_link.o: CMakeFiles/Reduction_DatasetGenerator.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/lclhome/tcickovs/PluMA/pipelines/GPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/Reduction_DatasetGenerator.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Reduction_DatasetGenerator.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Reduction_DatasetGenerator.dir/build: CMakeFiles/Reduction_DatasetGenerator.dir/cmake_device_link.o
.PHONY : CMakeFiles/Reduction_DatasetGenerator.dir/build

# Object files for target Reduction_DatasetGenerator
Reduction_DatasetGenerator_OBJECTS = \
"CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.o"

# External object files for target Reduction_DatasetGenerator
Reduction_DatasetGenerator_EXTERNAL_OBJECTS =

Reduction_DatasetGenerator: CMakeFiles/Reduction_DatasetGenerator.dir/Module9/Reduction/dataset_generator.cpp.o
Reduction_DatasetGenerator: CMakeFiles/Reduction_DatasetGenerator.dir/build.make
Reduction_DatasetGenerator: libgputk.a
Reduction_DatasetGenerator: /usr/local/cuda-11.2/lib64/libcudart.so
Reduction_DatasetGenerator: CMakeFiles/Reduction_DatasetGenerator.dir/cmake_device_link.o
Reduction_DatasetGenerator: CMakeFiles/Reduction_DatasetGenerator.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/lclhome/tcickovs/PluMA/pipelines/GPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable Reduction_DatasetGenerator"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Reduction_DatasetGenerator.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Reduction_DatasetGenerator.dir/build: Reduction_DatasetGenerator
.PHONY : CMakeFiles/Reduction_DatasetGenerator.dir/build

CMakeFiles/Reduction_DatasetGenerator.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Reduction_DatasetGenerator.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Reduction_DatasetGenerator.dir/clean

CMakeFiles/Reduction_DatasetGenerator.dir/depend:
	cd /lclhome/tcickovs/PluMA/pipelines/GPU/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /lclhome/tcickovs/PluMA/pipelines/GPU /lclhome/tcickovs/PluMA/pipelines/GPU /lclhome/tcickovs/PluMA/pipelines/GPU/build /lclhome/tcickovs/PluMA/pipelines/GPU/build /lclhome/tcickovs/PluMA/pipelines/GPU/build/CMakeFiles/Reduction_DatasetGenerator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Reduction_DatasetGenerator.dir/depend

