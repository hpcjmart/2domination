NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include  -lcublas_static -lculibos -lcudart_static -lpthread -ldl
NVCC_SPEC_FLAGS = --default-stream per-thread
LD_FLAGS    = -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lculibos -lcudart_static -lpthread -ldl
EXE1	    	= matrixTrop
OBJ1	    	= matrixTrop.o

default: $(EXE1)

1: $(EXE1)

all: $(EXE1)  

matrixTrop.o: matrixTrop.cu 
	$(NVCC) -c -o $@ matrixTrop.cu $(NVCC_FLAGS) 

$(EXE1): $(OBJ1)
	$(NVCC) $(OBJ1) -o $(EXE1) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE1)  
