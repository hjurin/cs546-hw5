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
float BLOCK_SIZE; /* Size of blocks */
float GRID_DIM; /* Size of the grid */

/* Arrays of mu and sigma for each column*/
volatile float M[MAXN], S[MAXN];

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
    if (argc >= 3) {
        seed = atoi(argv[2]);
        srand(seed);
        printf("Random seed = %i\n", seed);
    }
    GRID_DIM = N / ceil(argc >= 4 ? atof(argv[3]) : 8.0);
    printf("Grid size = %f\n", GRID_DIM);
    BLOCK_SIZE = ceil(argc >= 5 ? atof(argv[4]) : 8.0);
    printf("Blocks size = %f\n", BLOCK_SIZE);

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
        M[col] = 0.0;
        S[col] = 0.0;
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


/* Prototype of Kernel functions */
__global__ void muKernel(float * d_A, float * d_B, float * d_S, float * d_M, int size);
__global__ void sigmaKernel(float * d_A, float * d_B, float * d_S, float * d_M, int size);
__global__ void matrixNormKernel(float * d_A, float * d_B, float * d_S, float * d_M, int size);
void gaussianElimination();

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
    gaussianElimination();

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

void gaussianElimination() {
    // designed to be copies of global arrays
    float *d_A, *d_B, *d_M, *d_S;

    // allocation and copying of global arrays
    cudaMalloc((void**)&d_A, (N * N) * sizeof(float));
    cudaMalloc((void**)&d_B, (N * N) * sizeof(float));
    cudaMalloc((void**)&d_S, N * sizeof(float));
    cudaMalloc((void**)&d_M, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        cudaMemcpy(d_A + i * N, (float*)A[i], N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B + i * N, (float*)B[i], N * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_S, (float*)S, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, (float*)M, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch Kernel functions
    printf("Computing in parallel.\n");
    dim3 dimGrid(GRID_DIM, 1);
    dim3 dimBlock(BLOCK_SIZE, 1);
    // Computes mus for the whole matrix
    muKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_S, d_M, N);
    cudaMemcpy((float*)M, d_M, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nM =\n\t");
    for (int i = 0; i < N; i++) {
        printf("%f%s", M[i], (i < N-1) ? ", " : ";\n\t");
        M[i] /= (float)N;
    }
    cudaMemcpy(d_M, (float*)M, N * sizeof(float), cudaMemcpyHostToDevice);

    // Compute the sigmas for the whole matrix
    sigmaKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_S, d_M, N);
    cudaMemcpy((float*)S, d_S, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nS =\n\t");
    for (int i = 0; i < N; i++) {
        printf("%f%s", S[i], (i < N-1) ? ", " : ";\n\t");
        S[i] /= (float)N;
    }
    cudaMemcpy(d_S, (float*)S, N * sizeof(float), cudaMemcpyHostToDevice);

    // Filling of the normalized matrix
    matrixNormKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_S, d_M, N);

    // Copies back computed matrix
    for (int i = 0; i < N; i++) {
        cudaMemcpy((float*)B[i], d_B + i * N, N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Frees all arrays copies
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_S);
    cudaFree(d_M);
}

__global__ void muKernel(float * d_A, float * d_B, float * d_S, float * d_M, int size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row;

    // Thread workload
    for(row=0; row < size; row++) {
        if (col < size) {
            d_M[col] += d_A[row * size + col];
        }
    }
}

__global__ void sigmaKernel(float * d_A, float * d_B, float * d_S, float * d_M, int size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row;

    // Thread workload
    for(row=0; row < size; row++) {
        if (col < size) {
            d_S[col] += powf(d_A[row * size + col] - d_M[col], 2.0);
        }
    }
}

__global__ void matrixNormKernel(float * d_A, float * d_B, float * d_S, float * d_M, int size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row;

    // Thread workload
    for(row=0; row < size; row++) {
        if (d_S[col] == 0.0) {
            d_B[(row * size) + col] = 0.0;
        }
        else {
            d_B[row * size + col] = (d_A[row * size + col] - d_M[col]) / d_S[col];
        }
    }
}
