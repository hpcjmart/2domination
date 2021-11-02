#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
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

int loadfile_matrixMultiply(int argc, char **argv, 
		unsigned long matArow,unsigned long matAcol,unsigned long matBrow,unsigned long matBcol, 
		char *filename1, int iteraciones)
{


    struct timeval tv1, tv2, tv3;
    struct timezone tz;
    double elapsed;

    gettimeofday(&tv1, &tz);

    FILE *f1;
    f1=fopen(filename1,"r");

    size_t size_A = (unsigned long)(matArow) * (unsigned long)(matAcol);
    size_t size_B = (unsigned long)(matArow) * (unsigned long)(matBcol);
    short* h_A = (short *) malloc(size_A*sizeof(short));
    short* h_B = (short *) malloc(size_B*sizeof(short));

    size_t size_C = (unsigned long)(matArow) * (unsigned long)(matAcol);
    h_C = (short **)malloc(sizeof(short *)*iteraciones);
    for (unsigned int i = 0; i< iteraciones; i++ )
        h_C[i] = (short*) malloc(size_C*sizeof(short));

    short a;
    for (unsigned int i = 0; i < size_A; i++){
      fscanf(f1,"%d",&a);
      h_A[i] = a;
      h_B[i] = a;
      h_C[0][i] = 0.0;
    }
    fclose(f1);

    short Csub;

    unsigned int i,j,k;

    for(unsigned int it=0;it<iteraciones;it++){

        #pragma omp parallel for private(i,j,k,Csub) shared(h_A,h_B,h_C)
	for (i = 0; i < matAcol; ++i) {
        	for (j = 0; j < matBcol; ++j) {
            		Csub = 9999;
            		for (k = 0; k < matBcol; ++k) Csub = MIN(Csub,h_A[i*matAcol+k]+h_B[k*matBcol+j]);
                        h_C[it][i*matAcol+j] = Csub;
	 	}
        }

        for (unsigned int i = 0; i < size_B; i++) h_B[i]=h_C[it][i];


    }

    free(h_A);
    free(h_B);

    gettimeofday(&tv2, &tz);
    elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;

    printf("Time for %d iterations : %f s\n", iteraciones,elapsed);

    return 0;

}


int busqueda_recurrencia(unsigned long matArow,unsigned long matAcol,unsigned long matBrow,unsigned long matBcol,int iteraciones)
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

        for (unsigned int i = 0; i < size_C; i++){
                if(h_C[it1][i]>=9999) h_C[it1][i]=32767;
        }

	for(it2=it1-1;it2>=3;it2--){

	  #pragma omp parallel for private(i,j) shared(temp,h_C)
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

int busqueda_recurrencia_minima(unsigned long matArow,unsigned long matAcol,unsigned long matBrow,unsigned long matBcol)
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

	#pragma omp parallel for private(i,j) shared(temp,h_C)
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

    int iteraciones=atoi(argv[3]);
    int cores=atoi(argv[4]);

    omp_set_num_threads(cores);

    printf("Runnig %s on %d cores\n",argv[0],cores);

    int matrix_result = loadfile_matrixMultiply(argc, argv, matArow, matAcol, matBrow, matBcol, filename1, iteraciones);
    int recurrencia = busqueda_recurrencia(matArow, matAcol, matBrow, matBcol, iteraciones);
    int recurrencia_minima = busqueda_recurrencia_minima(matArow, matAcol, matBrow, matBcol);
    int diagonales = calculo_min_diagonales(matArow,matAcol,matBrow,matBcol);

    for(unsigned int i = 0; i< iteraciones; i++) free(h_C[i]);
    free(h_C);

    stopTime(&timer); printf("Total time : %f s\n", elapsedTime(timer));

    printf("r0m=%d n0m=%d am=%d bm=%d\n",r0m,n0m,am,(int)bm);

    exit(0);
}
