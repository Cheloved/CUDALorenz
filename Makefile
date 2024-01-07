# Folder with sources
SRC_FOLDER=src

#OBJS specifies which files to compile as part of the project
OBJS=*.cu

#CC specifies which compiler we're using
CC=nvcc

#LINKER_FLAGS specifies the libraries we're linking against
LINKER_FLAGS=-I /usr/local/cuda/targets/x86_64-linux/include -lGL -lGLU -lglut -lm -lcuda -lcudart

#OBJ_NAME specifies the name of our exectuable
OBJ_NAME=cudalorenz

#This is the target that compiles our executable
all: 
	$(CC) -dc $(SRC_FOLDER)/$(OBJS) # Compile separatly
	$(CC) *.o $(LINKER_FLAGS) -o $(OBJ_NAME)						# Link together
	rm *.o											# Remove object files

# Note:
# --device-c|-dc Compile each .c/.cc/.cpp/.cxx/.cu input file into an object file that contains relocatable device code.
