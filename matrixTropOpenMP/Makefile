GCC        = gcc
GCC_FLAGS  = -O3 -fopenmp
LD_FLAGS   = -fopenmp
EXE1	    	= matrixTropOpenMP
OBJ1	    	= matrixTropOpenMP.o

default: $(EXE1)

1: $(EXE1)

all: $(EXE1)  

matrixTropOpenMP.o: matrixTropOpenMP.c
	$(GCC) -c -o $@ matrixTropOpenMP.c $(GCC_FLAGS) 

$(EXE1): $(OBJ1)
	$(GCC) $(OBJ1) -o $(EXE1) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE1)  
