# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_SOURCE_DIR = /home/cjy/cuda-neural-network/cuda-neural-network-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cjy/cuda-neural-network/cuda-neural-network-master/build

# Include any dependencies generated for this target.
include CMakeFiles/cu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cu.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cu.dir/flags.make

CMakeFiles/cu.dir/src/cuda/blas.cu.o: CMakeFiles/cu.dir/flags.make
CMakeFiles/cu.dir/src/cuda/blas.cu.o: ../src/cuda/blas.cu
CMakeFiles/cu.dir/src/cuda/blas.cu.o: CMakeFiles/cu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cjy/cuda-neural-network/cuda-neural-network-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/cu.dir/src/cuda/blas.cu.o"
	/usr/local/cuda-12.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cu.dir/src/cuda/blas.cu.o -MF CMakeFiles/cu.dir/src/cuda/blas.cu.o.d -x cu -c /home/cjy/cuda-neural-network/cuda-neural-network-master/src/cuda/blas.cu -o CMakeFiles/cu.dir/src/cuda/blas.cu.o

CMakeFiles/cu.dir/src/cuda/blas.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cu.dir/src/cuda/blas.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cu.dir/src/cuda/blas.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cu.dir/src/cuda/blas.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cu.dir/src/cuda/conv.cu.o: CMakeFiles/cu.dir/flags.make
CMakeFiles/cu.dir/src/cuda/conv.cu.o: ../src/cuda/conv.cu
CMakeFiles/cu.dir/src/cuda/conv.cu.o: CMakeFiles/cu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cjy/cuda-neural-network/cuda-neural-network-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/cu.dir/src/cuda/conv.cu.o"
	/usr/local/cuda-12.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cu.dir/src/cuda/conv.cu.o -MF CMakeFiles/cu.dir/src/cuda/conv.cu.o.d -x cu -c /home/cjy/cuda-neural-network/cuda-neural-network-master/src/cuda/conv.cu -o CMakeFiles/cu.dir/src/cuda/conv.cu.o

CMakeFiles/cu.dir/src/cuda/conv.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cu.dir/src/cuda/conv.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cu.dir/src/cuda/conv.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cu.dir/src/cuda/conv.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cu.dir/src/cuda/flatten.cu.o: CMakeFiles/cu.dir/flags.make
CMakeFiles/cu.dir/src/cuda/flatten.cu.o: ../src/cuda/flatten.cu
CMakeFiles/cu.dir/src/cuda/flatten.cu.o: CMakeFiles/cu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cjy/cuda-neural-network/cuda-neural-network-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/cu.dir/src/cuda/flatten.cu.o"
	/usr/local/cuda-12.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cu.dir/src/cuda/flatten.cu.o -MF CMakeFiles/cu.dir/src/cuda/flatten.cu.o.d -x cu -c /home/cjy/cuda-neural-network/cuda-neural-network-master/src/cuda/flatten.cu -o CMakeFiles/cu.dir/src/cuda/flatten.cu.o

CMakeFiles/cu.dir/src/cuda/flatten.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cu.dir/src/cuda/flatten.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cu.dir/src/cuda/flatten.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cu.dir/src/cuda/flatten.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cu.dir/src/cuda/linear.cu.o: CMakeFiles/cu.dir/flags.make
CMakeFiles/cu.dir/src/cuda/linear.cu.o: ../src/cuda/linear.cu
CMakeFiles/cu.dir/src/cuda/linear.cu.o: CMakeFiles/cu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cjy/cuda-neural-network/cuda-neural-network-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/cu.dir/src/cuda/linear.cu.o"
	/usr/local/cuda-12.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cu.dir/src/cuda/linear.cu.o -MF CMakeFiles/cu.dir/src/cuda/linear.cu.o.d -x cu -c /home/cjy/cuda-neural-network/cuda-neural-network-master/src/cuda/linear.cu -o CMakeFiles/cu.dir/src/cuda/linear.cu.o

CMakeFiles/cu.dir/src/cuda/linear.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cu.dir/src/cuda/linear.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cu.dir/src/cuda/linear.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cu.dir/src/cuda/linear.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cu.dir/src/cuda/max_pool.cu.o: CMakeFiles/cu.dir/flags.make
CMakeFiles/cu.dir/src/cuda/max_pool.cu.o: ../src/cuda/max_pool.cu
CMakeFiles/cu.dir/src/cuda/max_pool.cu.o: CMakeFiles/cu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cjy/cuda-neural-network/cuda-neural-network-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object CMakeFiles/cu.dir/src/cuda/max_pool.cu.o"
	/usr/local/cuda-12.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cu.dir/src/cuda/max_pool.cu.o -MF CMakeFiles/cu.dir/src/cuda/max_pool.cu.o.d -x cu -c /home/cjy/cuda-neural-network/cuda-neural-network-master/src/cuda/max_pool.cu -o CMakeFiles/cu.dir/src/cuda/max_pool.cu.o

CMakeFiles/cu.dir/src/cuda/max_pool.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cu.dir/src/cuda/max_pool.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cu.dir/src/cuda/max_pool.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cu.dir/src/cuda/max_pool.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cu.dir/src/cuda/nll_loss.cu.o: CMakeFiles/cu.dir/flags.make
CMakeFiles/cu.dir/src/cuda/nll_loss.cu.o: ../src/cuda/nll_loss.cu
CMakeFiles/cu.dir/src/cuda/nll_loss.cu.o: CMakeFiles/cu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cjy/cuda-neural-network/cuda-neural-network-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CUDA object CMakeFiles/cu.dir/src/cuda/nll_loss.cu.o"
	/usr/local/cuda-12.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cu.dir/src/cuda/nll_loss.cu.o -MF CMakeFiles/cu.dir/src/cuda/nll_loss.cu.o.d -x cu -c /home/cjy/cuda-neural-network/cuda-neural-network-master/src/cuda/nll_loss.cu -o CMakeFiles/cu.dir/src/cuda/nll_loss.cu.o

CMakeFiles/cu.dir/src/cuda/nll_loss.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cu.dir/src/cuda/nll_loss.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cu.dir/src/cuda/nll_loss.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cu.dir/src/cuda/nll_loss.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cu.dir/src/cuda/relu.cu.o: CMakeFiles/cu.dir/flags.make
CMakeFiles/cu.dir/src/cuda/relu.cu.o: ../src/cuda/relu.cu
CMakeFiles/cu.dir/src/cuda/relu.cu.o: CMakeFiles/cu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cjy/cuda-neural-network/cuda-neural-network-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CUDA object CMakeFiles/cu.dir/src/cuda/relu.cu.o"
	/usr/local/cuda-12.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cu.dir/src/cuda/relu.cu.o -MF CMakeFiles/cu.dir/src/cuda/relu.cu.o.d -x cu -c /home/cjy/cuda-neural-network/cuda-neural-network-master/src/cuda/relu.cu -o CMakeFiles/cu.dir/src/cuda/relu.cu.o

CMakeFiles/cu.dir/src/cuda/relu.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cu.dir/src/cuda/relu.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cu.dir/src/cuda/relu.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cu.dir/src/cuda/relu.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cu.dir/src/cuda/rmsprop.cu.o: CMakeFiles/cu.dir/flags.make
CMakeFiles/cu.dir/src/cuda/rmsprop.cu.o: ../src/cuda/rmsprop.cu
CMakeFiles/cu.dir/src/cuda/rmsprop.cu.o: CMakeFiles/cu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cjy/cuda-neural-network/cuda-neural-network-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CUDA object CMakeFiles/cu.dir/src/cuda/rmsprop.cu.o"
	/usr/local/cuda-12.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cu.dir/src/cuda/rmsprop.cu.o -MF CMakeFiles/cu.dir/src/cuda/rmsprop.cu.o.d -x cu -c /home/cjy/cuda-neural-network/cuda-neural-network-master/src/cuda/rmsprop.cu -o CMakeFiles/cu.dir/src/cuda/rmsprop.cu.o

CMakeFiles/cu.dir/src/cuda/rmsprop.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cu.dir/src/cuda/rmsprop.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cu.dir/src/cuda/rmsprop.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cu.dir/src/cuda/rmsprop.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cu.dir/src/cuda/sigmoid.cu.o: CMakeFiles/cu.dir/flags.make
CMakeFiles/cu.dir/src/cuda/sigmoid.cu.o: ../src/cuda/sigmoid.cu
CMakeFiles/cu.dir/src/cuda/sigmoid.cu.o: CMakeFiles/cu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cjy/cuda-neural-network/cuda-neural-network-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CUDA object CMakeFiles/cu.dir/src/cuda/sigmoid.cu.o"
	/usr/local/cuda-12.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cu.dir/src/cuda/sigmoid.cu.o -MF CMakeFiles/cu.dir/src/cuda/sigmoid.cu.o.d -x cu -c /home/cjy/cuda-neural-network/cuda-neural-network-master/src/cuda/sigmoid.cu -o CMakeFiles/cu.dir/src/cuda/sigmoid.cu.o

CMakeFiles/cu.dir/src/cuda/sigmoid.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cu.dir/src/cuda/sigmoid.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cu.dir/src/cuda/sigmoid.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cu.dir/src/cuda/sigmoid.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cu.dir/src/cuda/softmax.cu.o: CMakeFiles/cu.dir/flags.make
CMakeFiles/cu.dir/src/cuda/softmax.cu.o: ../src/cuda/softmax.cu
CMakeFiles/cu.dir/src/cuda/softmax.cu.o: CMakeFiles/cu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cjy/cuda-neural-network/cuda-neural-network-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CUDA object CMakeFiles/cu.dir/src/cuda/softmax.cu.o"
	/usr/local/cuda-12.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cu.dir/src/cuda/softmax.cu.o -MF CMakeFiles/cu.dir/src/cuda/softmax.cu.o.d -x cu -c /home/cjy/cuda-neural-network/cuda-neural-network-master/src/cuda/softmax.cu -o CMakeFiles/cu.dir/src/cuda/softmax.cu.o

CMakeFiles/cu.dir/src/cuda/softmax.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cu.dir/src/cuda/softmax.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cu.dir/src/cuda/softmax.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cu.dir/src/cuda/softmax.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cu.dir/src/cuda/storage.cu.o: CMakeFiles/cu.dir/flags.make
CMakeFiles/cu.dir/src/cuda/storage.cu.o: ../src/cuda/storage.cu
CMakeFiles/cu.dir/src/cuda/storage.cu.o: CMakeFiles/cu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cjy/cuda-neural-network/cuda-neural-network-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CUDA object CMakeFiles/cu.dir/src/cuda/storage.cu.o"
	/usr/local/cuda-12.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cu.dir/src/cuda/storage.cu.o -MF CMakeFiles/cu.dir/src/cuda/storage.cu.o.d -x cu -c /home/cjy/cuda-neural-network/cuda-neural-network-master/src/cuda/storage.cu -o CMakeFiles/cu.dir/src/cuda/storage.cu.o

CMakeFiles/cu.dir/src/cuda/storage.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cu.dir/src/cuda/storage.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cu.dir/src/cuda/storage.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cu.dir/src/cuda/storage.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cu
cu_OBJECTS = \
"CMakeFiles/cu.dir/src/cuda/blas.cu.o" \
"CMakeFiles/cu.dir/src/cuda/conv.cu.o" \
"CMakeFiles/cu.dir/src/cuda/flatten.cu.o" \
"CMakeFiles/cu.dir/src/cuda/linear.cu.o" \
"CMakeFiles/cu.dir/src/cuda/max_pool.cu.o" \
"CMakeFiles/cu.dir/src/cuda/nll_loss.cu.o" \
"CMakeFiles/cu.dir/src/cuda/relu.cu.o" \
"CMakeFiles/cu.dir/src/cuda/rmsprop.cu.o" \
"CMakeFiles/cu.dir/src/cuda/sigmoid.cu.o" \
"CMakeFiles/cu.dir/src/cuda/softmax.cu.o" \
"CMakeFiles/cu.dir/src/cuda/storage.cu.o"

# External object files for target cu
cu_EXTERNAL_OBJECTS =

libcu.a: CMakeFiles/cu.dir/src/cuda/blas.cu.o
libcu.a: CMakeFiles/cu.dir/src/cuda/conv.cu.o
libcu.a: CMakeFiles/cu.dir/src/cuda/flatten.cu.o
libcu.a: CMakeFiles/cu.dir/src/cuda/linear.cu.o
libcu.a: CMakeFiles/cu.dir/src/cuda/max_pool.cu.o
libcu.a: CMakeFiles/cu.dir/src/cuda/nll_loss.cu.o
libcu.a: CMakeFiles/cu.dir/src/cuda/relu.cu.o
libcu.a: CMakeFiles/cu.dir/src/cuda/rmsprop.cu.o
libcu.a: CMakeFiles/cu.dir/src/cuda/sigmoid.cu.o
libcu.a: CMakeFiles/cu.dir/src/cuda/softmax.cu.o
libcu.a: CMakeFiles/cu.dir/src/cuda/storage.cu.o
libcu.a: CMakeFiles/cu.dir/build.make
libcu.a: CMakeFiles/cu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cjy/cuda-neural-network/cuda-neural-network-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Linking CUDA static library libcu.a"
	$(CMAKE_COMMAND) -P CMakeFiles/cu.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cu.dir/build: libcu.a
.PHONY : CMakeFiles/cu.dir/build

CMakeFiles/cu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cu.dir/clean

CMakeFiles/cu.dir/depend:
	cd /home/cjy/cuda-neural-network/cuda-neural-network-master/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cjy/cuda-neural-network/cuda-neural-network-master /home/cjy/cuda-neural-network/cuda-neural-network-master /home/cjy/cuda-neural-network/cuda-neural-network-master/build /home/cjy/cuda-neural-network/cuda-neural-network-master/build /home/cjy/cuda-neural-network/cuda-neural-network-master/build/CMakeFiles/cu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cu.dir/depend

