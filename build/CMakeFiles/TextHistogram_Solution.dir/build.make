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
include CMakeFiles/TextHistogram_Solution.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/TextHistogram_Solution.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/TextHistogram_Solution.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/TextHistogram_Solution.dir/flags.make

CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.o: CMakeFiles/TextHistogram_Solution.dir/flags.make
CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.o: ../Module7/TextHistogram/solution.cu
CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.o: CMakeFiles/TextHistogram_Solution.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/lclhome/tcickovs/PluMA/pipelines/GPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.o"
	/usr/local/cuda-11.2/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.o -MF CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.o.d -x cu -dc /lclhome/tcickovs/PluMA/pipelines/GPU/Module7/TextHistogram/solution.cu -o CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.o

CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target TextHistogram_Solution
TextHistogram_Solution_OBJECTS = \
"CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.o"

# External object files for target TextHistogram_Solution
TextHistogram_Solution_EXTERNAL_OBJECTS =

CMakeFiles/TextHistogram_Solution.dir/cmake_device_link.o: CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.o
CMakeFiles/TextHistogram_Solution.dir/cmake_device_link.o: CMakeFiles/TextHistogram_Solution.dir/build.make
CMakeFiles/TextHistogram_Solution.dir/cmake_device_link.o: libgputk.a
CMakeFiles/TextHistogram_Solution.dir/cmake_device_link.o: /usr/local/cuda-11.2/lib64/libcudart.so
CMakeFiles/TextHistogram_Solution.dir/cmake_device_link.o: CMakeFiles/TextHistogram_Solution.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/lclhome/tcickovs/PluMA/pipelines/GPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/TextHistogram_Solution.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TextHistogram_Solution.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/TextHistogram_Solution.dir/build: CMakeFiles/TextHistogram_Solution.dir/cmake_device_link.o
.PHONY : CMakeFiles/TextHistogram_Solution.dir/build

# Object files for target TextHistogram_Solution
TextHistogram_Solution_OBJECTS = \
"CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.o"

# External object files for target TextHistogram_Solution
TextHistogram_Solution_EXTERNAL_OBJECTS =

TextHistogram_Solution: CMakeFiles/TextHistogram_Solution.dir/Module7/TextHistogram/solution.cu.o
TextHistogram_Solution: CMakeFiles/TextHistogram_Solution.dir/build.make
TextHistogram_Solution: libgputk.a
TextHistogram_Solution: /usr/local/cuda-11.2/lib64/libcudart.so
TextHistogram_Solution: CMakeFiles/TextHistogram_Solution.dir/cmake_device_link.o
TextHistogram_Solution: CMakeFiles/TextHistogram_Solution.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/lclhome/tcickovs/PluMA/pipelines/GPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable TextHistogram_Solution"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TextHistogram_Solution.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/TextHistogram_Solution.dir/build: TextHistogram_Solution
.PHONY : CMakeFiles/TextHistogram_Solution.dir/build

CMakeFiles/TextHistogram_Solution.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/TextHistogram_Solution.dir/cmake_clean.cmake
.PHONY : CMakeFiles/TextHistogram_Solution.dir/clean

CMakeFiles/TextHistogram_Solution.dir/depend:
	cd /lclhome/tcickovs/PluMA/pipelines/GPU/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /lclhome/tcickovs/PluMA/pipelines/GPU /lclhome/tcickovs/PluMA/pipelines/GPU /lclhome/tcickovs/PluMA/pipelines/GPU/build /lclhome/tcickovs/PluMA/pipelines/GPU/build /lclhome/tcickovs/PluMA/pipelines/GPU/build/CMakeFiles/TextHistogram_Solution.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/TextHistogram_Solution.dir/depend

