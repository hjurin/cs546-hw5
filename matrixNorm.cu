/* Matrix normalization using CUDA
* Compile with "nvcc matrixNorm.cu"
*/

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
* You need not submit the provided code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

/* Program Parameters */
#define MAXN 8000  /* Max value of N */
int N;  /* Matrix size */

/* Matrices */
volatile float A[MAXN][MAXN], B[MAXN][MAXN];

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void matrixNorm();

/* returns a seed for srand based on the time */
unsigned int time_seed() {
    struct timeval t;
    struct timezone tzdummy;

    gettimeofday(&t, &tzdummy);
    return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
    int seed = 0;  /* Random seed */

    /* Read command-line arguments */
    srand(time_seed());  /* Randomize */

    if (argc == 3) {
        seed = atoi(argv[2]);
        srand(seed);
        printf("Random seed = %i\n", seed);
    }
    if (argc >= 2) {
        N = atoi(argv[1]);
        if (N < 1 || N > MAXN) {
            printf("N = %i is out of range.\n", N);
            exit(0);
        }
    }
    else {
        printf("Usage: %s <matrix_dimension> [random seed]\n",
        argv[0]);
        exit(0);
    }

    /* Print parameters */
    printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B*/
void initialize_inputs() {
    int row, col;

    printf("\nInitializing...\n");

    for (col = 0; col < N; col++) {
        for (row = 0; row < N; row++) {
            A[row][col] = (float)rand() / 32768.0;
            B[row][col] = 0.0;
        }
    }
    /*
    for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
    A[row][col] = col + row;
    B[row][col] = 0.0;
}
}
*/

}

/* Print input matrices */
void print_inputs() {
    int row, col;

    if (N < 10) {
        printf("\nA =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }
    }
}

void print_B() {
    int row, col;

    if (N < 10) {
        printf("\nB =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%1.10f%s", B[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }
    }
}


/* Prototype of the Kernel function */
__global__ void matrixNormKernel(float * d_A, float * d_B, int size);

int main(int argc, char **argv) {
    /* Timing variables */
    struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
    struct timezone tzdummy;
    unsigned long long usecstart, usecstop;
    struct tms cputstart, cputstop;  /* CPU times for my processes */

    /* Process program parameters */
    parameters(argc, argv);

    /* Initialize A and B */
    initialize_inputs();

    /* Print input matrices */
    print_inputs();

    /* Start Clock */
    printf("\nStarting clock.\n");
    gettimeofday(&etstart, &tzdummy);
    times(&cputstart);

    /* Gaussian Elimination */
    float *d_A, *d_B;

    cudaMalloc((void**)&d_A, (N * N) * sizeof(float));
    cudaMalloc((void**)&d_B, (N * N) * sizeof(float));
    for (int i = 0; i < N; i++) {
        cudaMemcpy(d_A + i * N, (float*)A[i], N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B + i * N, (float*)B[i], N * sizeof(float), cudaMemcpyHostToDevice);
    }

    dim3 dimGrid(ceil(N/8.0), 1);
    dim3 dimBlock(8, 1);
    printf("Computing Serially.\n");
    matrixNormKernel<<<dimGrid, dimBlock>>>(d_A, d_B, N);
    for (int i = 0; i < N; i++) {
        cudaMemcpy((float*)A[i], d_A + i * N, (N * N) * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy((float*)B[i], d_B + i * N, (N * N) * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_A);
    cudaFree(d_B);
    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    times(&cputstop);
    printf("Stopped clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

    /* Display output */
    print_B();

    /* Display timing results */
    printf("\nElapsed time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);

    printf("(CPU times are accurate to the nearest %g ms)\n",
    1.0/(float)CLOCKS_PER_SEC * 1000.0);
    printf("My total CPU time for parent = %g ms.\n",
    (float)( (cputstop.tms_utime + cputstop.tms_stime) -
    (cputstart.tms_utime + cputstart.tms_stime) ) /
    (float)CLOCKS_PER_SEC * 1000);
    printf("My system CPU time for parent = %g ms.\n",
    (float)(cputstop.tms_stime - cputstart.tms_stime) /
    (float)CLOCKS_PER_SEC * 1000);
    printf("My total CPU time for child processes = %g ms.\n",
    (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
    (cputstart.tms_cutime + cputstart.tms_cstime) ) /
    (float)CLOCKS_PER_SEC * 1000);
    /* Contrary to the man pages, this appears not to include the parent */
    printf("--------------------------------------------\n");

    exit(0);
}

/* ------------------ Above Was Provided --------------------- */

__global__ void matrixNormKernel(float * d_A, float * d_B, int size) {
    int tx = threadIdx.x;
    int bd = blockDim.x;
    int bx = blockIdx.x;
    int row;
    float mu, sigma;

    // Thread workload
    mu = 0.0;
    for(row=0; row < size; row++) {
        if (bx * bd + tx < size) {
            mu += d_A[(row * size) + (bx * bd + tx)];
        }
    }
    mu /= (float) size;
    sigma = 0.0;
    for(row=0; row < size; row++) {
        sigma += powf(d_A[(row * size) + (bx * bd + tx)] - mu, 2.0);
    }
    sigma /= (float) size;
    for(row=0; row < size; row++) {
        if (sigma == 0.0) {
            d_B[(row * size) + (bx * bd + tx)] = 0.0;
        }
        else {
            d_B[(row * size) + (bx * bd + tx)] = (d_A[(row * size) + (bx * bd + tx)] - mu) / sigma;
        }
    }
}
