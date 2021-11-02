#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define MIN(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _b : _a; })


typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

short **h_C;

short bm;
int r0m;
int am;
short bm2;
int n0m;

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}


template <int BLOCK_SIZE> __global__ void matrixTropCUDA(short *C, short *A, short *B, int wA, int wB)
{
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int aBegin = wA * BLOCK_SIZE * by;
    unsigned int aEnd   = aBegin + wA - 1;
    unsigned int aStep  = BLOCK_SIZE;
    unsigned int bBegin = BLOCK_SIZE * bx;
    unsigned int bStep  = BLOCK_SIZE * wB;
    short Csub = 9999;
    for (unsigned int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        __shared__ short As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ short Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();

#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub = MIN( Csub, (As[ty][k] + Bs[k][tx]) );
        }

        __syncthreads();
    }

    unsigned int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

int loadfile_matrixMultiply(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB, char *filename1, int iteraciones)
{


    FILE *f1;
    f1=fopen(filename1,"r");

    Timer timer;

    startTime(&timer);

    size_t size_A = (unsigned long)(dimsA.x) * (unsigned long)(dimsA.y);
    size_t size_B = (unsigned long)(dimsB.x) * (unsigned long)(dimsB.y);
    short* h_A = (short *) malloc(size_A*sizeof(short));
    short* h_B = (short *) malloc(size_B*sizeof(short));

    dim3 dimsC(dimsB.x, dimsA.y, 1);
    size_t size_C = (unsigned long)(dimsC.x) * (unsigned long)(dimsC.y);
    h_C = (short **)malloc(sizeof(short *)*iteraciones);
    for (unsigned int i = 0; i< iteraciones; i++ )
        h_C[i] = (short*) malloc(size_C*sizeof(short));

    short* temp = (short*) malloc(size_C*sizeof(short));

    short a;
    for (unsigned int i = 0; i < size_A; i++){
      fscanf(f1,"%d",&a);
      h_A[i] = a;
      h_B[i] = a;
      h_C[0][i] = 0.0;
    }
    fclose(f1);

    short* d_A=0;
    short* d_B=0;
    short* d_C=0;

    cudaError_t error;

    error = cudaMalloc((void **) &d_A, size_A*sizeof(short));
    error = cudaMalloc((void **) &d_B, size_B*sizeof(short));
    error = cudaMalloc((void **) &d_C, size_C*sizeof(short));

    error = cudaMemcpy(d_A, h_A, size_A*sizeof(short), cudaMemcpyHostToDevice);
    error = cudaMemcpy(d_B, h_B, size_B*sizeof(short), cudaMemcpyHostToDevice);

    int devID = 0;
    cudaDeviceProp deviceProp;
    error = cudaGetDeviceProperties(&deviceProp, devID);
    error = cudaSetDevice(0);

    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    cudaDeviceSynchronize();

    for(unsigned int it=0;it<iteraciones;it++){

        matrixTropCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x); //2

	cudaDeviceSynchronize();
	error = cudaMemcpy(temp, d_C, size_C*sizeof(short), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	for (unsigned int i = 0; i < size_C; i++){
		h_C[it][i] = temp[i];
	}

	error = cudaMemcpy(d_B, d_C, size_C*sizeof(short), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

    }

    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    stopTime(&timer); printf("Time for %d iterations : %f s\n", iteraciones,elapsedTime(timer));

    return 0;

}

__global__ void
vectorAdd(short *A, short *B, short *C, size_t size)
{
    unsigned long i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
    {
        C[i] = A[i] - B[i];
    }
}

template <int BLOCK_SIZE> __global__ void matrixDiffCUDA(short *C, short *A, short *B, int wA, int wB)
{

    unsigned int col = BLOCK_SIZE*blockIdx.x+threadIdx.x ;
    unsigned int row = BLOCK_SIZE*blockIdx.y+threadIdx.y ;


    C[row*wB+col] = A[row*wA+col] - B[row*wB+col];

}

int busqueda_recurrencia(int block_size, dim3 &dimsA, dim3 &dimsB,int iteraciones)
{


    Timer timer;

    startTime(&timer);

    dim3 dimsC(dimsB.x, dimsA.y, 1);
    size_t size_C = (unsigned long)(dimsC.x) * (unsigned long)(dimsC.y);
    short* temp = (short*) malloc(size_C*sizeof(short));


    short* d_A=0;
    short* d_B=0;
    short* d_C=0;

    cudaError_t error;

    error = cudaMalloc((void **) &d_A, size_C*sizeof(short));
    error = cudaMalloc((void **) &d_B, size_C*sizeof(short));
    error = cudaMalloc((void **) &d_C, size_C*sizeof(short));

    int devID = 0;
    cudaDeviceProp deviceProp;
    error = cudaGetDeviceProperties(&deviceProp, devID);
    error = cudaSetDevice(0);

    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);


    int it1,it2;
    int salida;
    for(it1=iteraciones-1;it1>=3;it1--){

        for (unsigned int i = 0; i < size_C; i++){
                if(h_C[it1][i]>=9999) temp[i]=32767;
                else temp[i]=h_C[it1][i];
        }

        error = cudaMemcpy(d_A,temp,size_C*sizeof(short), cudaMemcpyHostToDevice);

	for(it2=it1-1;it2>=3;it2--){

          for (unsigned long i = 0; i < size_C; i++){
                temp[i]=h_C[it2][i];
          }

          error = cudaMemcpy(d_B,temp,size_C*sizeof(short), cudaMemcpyHostToDevice);

	  matrixDiffCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);

          cudaDeviceSynchronize();

          error = cudaMemcpy(temp, d_C, size_C*sizeof(short), cudaMemcpyDeviceToHost);
          cudaDeviceSynchronize();


          bm=temp[0];
          salida=0;
          for (unsigned long i = 1; i < size_C; i++){
           if(temp[i]>=22768) {salida=0;continue;}
           if(bm!=temp[i]) {salida=1;break;}
          }
          if (salida==0) {
		  break;
          }
       }
       if (salida==0) break;
    }

    am=it1-it2;
    r0m=it2+2;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(temp);
   
    stopTime(&timer); printf("Search recurrence time : %f s\n", elapsedTime(timer));

    return 0;
}

int busqueda_recurrencia_minima(int block_size, dim3 &dimsA, dim3 &dimsB)
{
    Timer timer;

    startTime(&timer);

    dim3 dimsC(dimsB.x, dimsA.y, 1);
    size_t size_C = (unsigned long)(dimsC.x) * (unsigned long)(dimsC.y);
    short* temp = (short*) malloc(size_C*sizeof(short));

    short* d_A=0;
    short* d_B=0;
    short* d_C=0;

    cudaError_t error;

    int salida;

    error = cudaMalloc((void **) &d_A, size_C*sizeof(short));
    error = cudaMalloc((void **) &d_B, size_C*sizeof(short));
    error = cudaMalloc((void **) &d_C, size_C*sizeof(short));

    int devID = 0;
    cudaDeviceProp deviceProp;
    error = cudaGetDeviceProperties(&deviceProp, devID);
    error = cudaSetDevice(0);

    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
 
    for(unsigned long ii=((r0m+am)-2);ii>=am;ii--){

        for (unsigned int i = 0; i < size_C; i++){
                if(h_C[ii][i]>=9999) temp[i]=32767;
                else temp[i]=h_C[ii][i];
        }

        error = cudaMemcpy(d_A,temp,size_C*sizeof(short), cudaMemcpyHostToDevice);

        for (unsigned long i = 0; i < size_C; i++){
                temp[i]=h_C[(ii-am)][i];
        }

        error = cudaMemcpy(d_B,temp,size_C*sizeof(short), cudaMemcpyHostToDevice);

        matrixDiffCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);

        cudaDeviceSynchronize();

        error = cudaMemcpy(temp, d_C, size_C*sizeof(short), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        salida=0;

	bm2=temp[0];
        for (unsigned long i = 1; i < size_C; i++){
           if(temp[i]>=22768) {salida=0;continue;}
           if(bm2!=temp[i]) {n0m=(ii-am)+1+2;salida=1;break;}
        }
        if (salida==1) break;

    }

    free(temp);

    stopTime(&timer); printf("Search minimal recurrence time : %f s\n", elapsedTime(timer));

    return 0;
}

int calculo_min_diagonales(unsigned long matArow,unsigned long matAcol,unsigned long matBrow,unsigned long matBcol)  
{
    Timer timer;

    startTime(&timer);

    short menor;
    size_t size_A = (unsigned long)(matArow) * (unsigned long)(matAcol);

    for (unsigned long i = 3; i <= (n0m+am-1); i++){

      menor=32767;
      for(unsigned long j=0;j<size_A;j=(j+matArow+1))
	    if(h_C[i-2][j]<menor) menor=h_C[i-2][j];

      printf("Potencia %i | Menor %d\n",i,(int)menor);

    } 

    stopTime(&timer); printf("Minimal diagonal time : %f s\n", elapsedTime(timer));

    return 0;
}

int busqueda_recurrencia_CPU(unsigned long matArow,unsigned long matAcol,unsigned long matBrow,unsigned long matBcol,int iteraciones)
{


    Timer timer;

    startTime(&timer);

    size_t size_A = (unsigned long)(matArow) * (unsigned long)(matAcol);
    size_t size_B = (unsigned long)(matArow) * (unsigned long)(matBcol);
    short* h_A = (short *) malloc(size_A*sizeof(short));
    short* h_B = (short *) malloc(size_B*sizeof(short));

    size_t size_C = (unsigned long)(matArow) * (unsigned long)(matAcol);
    short* temp = (short*) malloc(size_C*sizeof(short));

    int it1,it2;
    int salida;
    unsigned int i,j;

    for(it1=iteraciones-1;it1>=3;it1--){

        for (i = 0; i < size_C; i++){
                if(h_C[it1][i]>=9999) h_C[it1][i]=32767;
        }

        for(it2=it1-1;it2>=3;it2--){

          for (i = 0; i < matAcol; ++i) {
                for (j = 0; j < matArow; ++j) {
                        temp[i*matAcol+j]=h_C[it1][i*matAcol+j]-h_C[it2][i*matAcol+j];
                }
          }

          bm=temp[0];
          salida=0;
          for (i = 1; i < size_C; i++){
           if(temp[i]>=22768) {salida=0;continue;}
           if(bm!=temp[i]) {salida=1;break;}
          }
          if (salida==0) {
                  break;
          }
       }
       if (salida==0) break;
    }

    am=it1-it2;
    r0m=it2+2;

    free(temp);
   
    stopTime(&timer); printf("Search recurrence time : %f s\n", elapsedTime(timer));

    return 0;
}

int busqueda_recurrencia_minima_CPU(unsigned long matArow,unsigned long matAcol,unsigned long matBrow,unsigned long matBcol)
{
    Timer timer;

    startTime(&timer);

    size_t size_C = (unsigned long)(matArow) * (unsigned long)(matAcol);
    short* temp = (short*) malloc(size_C*sizeof(short));

    int salida;
    unsigned int i,j;

    for(unsigned long ii=((r0m+am)-2);ii>=am;ii--){

        for (i = 0; i < size_C; i++){
                if(h_C[ii][i]>=9999) h_C[ii][i]=32767;
        }

        for (i = 0; i < matAcol; ++i) {
                for (j = 0; j < matArow; ++j) {
                        temp[i*matAcol+j]=h_C[ii][i*matAcol+j]-h_C[(ii-am)][i*matAcol+j];
                }
        }

        salida=0;

        bm2=temp[0];
        for (i = 1; i < size_C; i++){
           if(temp[i]>=22768) {salida=0;continue;}
           if(bm2!=temp[i]) {n0m=(ii-am)+1+2;salida=1;break;}
        }
        if (salida==1) break;

    }

    free(temp);

    stopTime(&timer); printf("Search minimal recurrence time : %f s\n", elapsedTime(timer));

    return 0;
}


int main(int argc, char **argv)
{

    Timer timer;

    startTime(&timer);

    char *filename1=argv[1];;

    unsigned long matArow, matAcol;
    unsigned long matBrow, matBcol;

    matArow=atoi(argv[2]);
    matAcol=atoi(argv[2]);
    matBrow=atoi(argv[2]);
    matBcol=atoi(argv[2]);

    int block_size = 32;

    dim3 dimsA(5*2*block_size, 5*2*block_size, 1);
    dim3 dimsB(5*4*block_size, 5*2*block_size, 1);

    int iteraciones=atoi(argv[3]);

    dimsA.x = matAcol;
    dimsA.y = matArow;
    dimsB.x = matBcol;
    dimsB.y = matBrow;


    int matrix_result = loadfile_matrixMultiply(argc, argv, block_size, dimsA, dimsB, filename1, iteraciones);
    int recurrencia = busqueda_recurrencia_CPU(matArow, matAcol, matBrow, matBcol, iteraciones);
    int recurrencia_minima = busqueda_recurrencia_minima_CPU(matArow, matAcol, matBrow, matBcol);
    int diagonales = calculo_min_diagonales(matArow,matAcol,matBrow,matBcol);

    for(unsigned int i = 0; i< iteraciones; i++) free(h_C[i]);
    free(h_C);

    stopTime(&timer); printf("Total time : %f s\n", elapsedTime(timer));

    printf("r0m=%d n0m=%d am=%d bm=%d\n",r0m,n0m,am,(int)bm);

    exit(0);
}
