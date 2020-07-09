#include <iostream>
#include <cmath>
#include <ctime>
#include <fstream>
#include <sstream>
#include <thrust/reduce.h>
#include <cusolverDn.h>
#include <time.h>

double* read_datasets(const std::string file_name, int num_rows, int num_cols) {

    std::ifstream file(file_name);
    std::string line = "";

    int r = 0, c = 0;
    double *A = new double[num_rows*num_cols];

    while (getline(file, line) && r < num_rows) {
        std::istringstream ss(line);

        for (double i; ss >> i && c < num_cols;) {
            A[r * num_cols + c] = i;
            c++;
            if (ss.peek() == ',')
                ss.ignore();
        }

        c = 0;
        r++;
    }

    file.close();

    return A;
}

class EPsim {

    private:
        int ASSETS;
        double *g_alpha, *g_sigma;
        double* initIdentityMatrix(const int NUM_ROWS, const int NUM_COLS);
        void matrixInverse(double* A, int n);

    public:
        EPsim(double* sigma, double* alpha, const int _n);
        void solve(int m, double g_a, double g_b);
        ~EPsim();
};

EPsim::EPsim(double* sigma, double* alpha, const int _n) {

    ASSETS = _n; 

    size_t matrix_size = ASSETS * ASSETS * sizeof(double);
    size_t vector_size = ASSETS * sizeof(double);

    cudaMallocManaged(&g_sigma, matrix_size);
    cudaMallocManaged(&g_alpha, vector_size);

    memcpy(g_sigma, sigma, matrix_size);
    memcpy(g_alpha, alpha, vector_size);

}

__global__ void gpuInitIdentity(double* d_A, const int NUM_ROWS, const int NUM_COLS) {

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < NUM_ROWS && x < NUM_COLS) {
        int idx = y * NUM_COLS + x;

        if (x == y) 
            d_A[idx] = 1;
        else
            d_A[idx] = 0;
    }

}

double* EPsim::initIdentityMatrix(const int NUM_ROWS, const int NUM_COLS) {
    
    double* d_A;
    cudaMallocManaged(&d_A, NUM_ROWS*NUM_COLS*sizeof(double));

    dim3 dimBlocks(32, 32);
    dim3 dimGrids(
            (NUM_COLS+dimBlocks.x-1)/dimBlocks.x,
            (NUM_ROWS+dimBlocks.y-1)/dimBlocks.y
    );
    
    gpuInitIdentity<<<dimGrids, dimBlocks>>>(d_A, NUM_ROWS, NUM_COLS);
    cudaDeviceSynchronize();

    return d_A;
}


void EPsim::matrixInverse(double* d_A, int n) {

    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    int lwork = 0;
    int *d_Ipiv, *d_info;
    double *d_work = NULL;

    cudaMalloc((void**)&d_Ipiv, n*sizeof(int));
    cudaMalloc((void**)&d_info, sizeof(int));

    cusolverDnDgetrf_bufferSize(cusolverH, n, n, d_A, n, &lwork);
    cudaMalloc((void**)&d_work, lwork*sizeof(double));

    // P * A = L * U
    cusolverDnDgetrf(cusolverH, n, n, d_A, n, d_work, d_Ipiv, d_info);
    cudaDeviceSynchronize();

    double* d_B = initIdentityMatrix(n, n);

    // LUx = B; 
    cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, n, n, d_A, n, d_Ipiv, d_B, n, d_info);
    cudaDeviceSynchronize();

    memcpy(d_A, d_B, n*n*sizeof(double));
    cudaDeviceSynchronize();

    cudaFree(d_Ipiv);
    cudaFree(d_info);
    cudaFree(d_work);
    cudaFree(d_B);
    cusolverDnDestroy(cusolverH);
}

__global__ void computePhase1(double* g_sigmaI, double* g_alpha, 
                              double* g_t1, double* g_t2,
                              double* g_p, double* g_q, double* g_r, 
                              int ASSETS) {

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < ASSETS) {

        double s1 = 0;
        double s2 = 0;

        for (int i = 0; i < ASSETS; ++i) {
            int r = (tid * ASSETS) + i;
            s1 += g_sigmaI[r] * g_alpha[i];
            s2 += g_sigmaI[r];
        }

        g_t1[tid] = s1;
        g_t2[tid] = s2;

        g_p[tid] = g_alpha[tid] * g_t1[tid];
        g_q[tid] = g_alpha[tid] * g_t2[tid];
        g_r[tid] = g_t2[tid];

    }
}

__global__ void gpuComputeEP(double* g_W0, double* g_M,
                              double* g_t1, double* g_t2,
                              double g_a, double g_b,
                              double g_alpha_l, double g_alpha_h,
                              int g_m, double g_n) {
                              
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (t_i < g_m) {
        double t_alpha_a = g_alpha_l + g_a * (g_alpha_h - g_alpha_l);
        double t_alpha_b = g_alpha_l + g_b * (g_alpha_h - g_alpha_l);

        double t_delta = (t_alpha_b - t_alpha_a) / ((double)g_m - 1.0);
        double t_alpha = t_alpha_a + (t_delta * t_i);

        double t_lambda1 = (t_alpha * g_M[0]) + g_M[1];
        double t_lambda2 = (t_alpha * g_M[2]) + g_M[3];

        for (int j = 0; j < g_n; ++j) {
            int r = (t_i* g_n) + j;
            g_W0[r] = t_lambda1 * g_t1[j] + t_lambda2 * g_t2[j];
        }
    }

}

void EPsim::solve(int m, double g_a, double g_b) {

    size_t vector_size = ASSETS * sizeof(double);

    double *g_t1, *g_t2; 
    double *g_p, *g_q, *g_r;

    cudaMallocManaged(&g_t1, vector_size);
    cudaMallocManaged(&g_t2, vector_size);
    cudaMallocManaged(&g_p, vector_size);
    cudaMallocManaged(&g_q, vector_size);
    cudaMallocManaged(&g_r, vector_size);

    int NUM_BLOCKS = 1;
    int NUM_THREADS = ASSETS;
    
    if (ASSETS > 1024) {
        NUM_BLOCKS = ceil((double)ASSETS/1024.0);
        NUM_THREADS = 1024;
    }

    /* first phase */
    matrixInverse(g_sigma, ASSETS); // sigmaI
    computePhase1<<<NUM_BLOCKS, NUM_THREADS>>>(
                                                g_sigma, g_alpha,
                                                g_t1, g_t2,
                                                g_p, g_q, g_r,
                                                ASSETS
                                            );
    cudaDeviceSynchronize();

    double *g_M; 
    cudaMallocManaged(&g_M, 4*sizeof(double));

    g_M[0] = thrust::reduce(thrust::device, g_p, g_p+ASSETS); 
    g_M[1] = g_M[2] = thrust::reduce(thrust::device, g_q, g_q+ASSETS); 
    g_M[3] = thrust::reduce(thrust::device, g_r, g_r+ASSETS); 

    matrixInverse(g_M, 2);

    if (m > 1024) {
        NUM_BLOCKS = ceil((double)m/1024.0);
        NUM_THREADS = 1024;
    } else {
        NUM_BLOCKS = 1;
        NUM_THREADS = m;
    }

    /* second phase */
    double g_alpha_h = *thrust::max_element(thrust::device, g_alpha, g_alpha+ASSETS);
    double g_alpha_l = *thrust::min_element(thrust::device, g_alpha, g_alpha+ASSETS);

    double *g_W0;
    cudaMallocManaged(&g_W0, m*ASSETS*sizeof(double));

    gpuComputeEP<<<NUM_BLOCKS, NUM_THREADS>>>(
                                            g_W0, g_M,
                                            g_t1, g_t2,
                                            g_a, g_b,
                                            g_alpha_l, g_alpha_h,
                                            m, ASSETS 
                                        ); 
    cudaDeviceSynchronize();
}

int main() {

    //int ASSETS = N;
    //EPsim *epsim = new EPsim(cov, mean, ASSETS);

    int m = 10000;
    int ASSETS = 1200;

    double* g_sigma = read_datasets("sigma4144x4144.csv", ASSETS, ASSETS);
    double* g_alpha = read_datasets("alpha4144.csv", ASSETS, 1);

    EPsim *epsim = new EPsim(g_sigma, g_alpha, ASSETS);

    double g_a = 0.0;
    double g_b = 1.0;

    clock_t begin = clock();

    epsim -> solve(m, g_a, g_b);

    clock_t end = clock();
    double elapsed_secs = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("[ASSETS = %d, m = %d] -> time = %.10lf\n", ASSETS, m, elapsed_secs);


    return 0;
}
