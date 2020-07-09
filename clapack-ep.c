#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include "cblas.h"
#include "clapack.h"
#include <time.h>

double* read_datasets(char* file_name, int num_rows, int num_cols) {

    FILE* fp = fopen(file_name, "r");

    char line[131072]; // 2**17
    double* A = (double*)malloc(num_rows*num_cols*sizeof(double));

    int idx = 0;
    int r = 0, c = 0;

    while (fgets(line, 131072, fp) && r < num_rows) {

        char* tmp = strdup(line);
        const char* token;
        
        for (token = strtok(line, ","); 
             token && *token && (c < num_cols);
             token = strtok(NULL, ",\n")) 
        {

            A[idx] = atof(token);

            idx++;
            c++;
        }

        c = 0;
        r++;

        free(tmp);
    }

    fclose(fp);

    return A;
}

double* initOnes(int n) {

    double *ones = (double*)malloc(n*sizeof(double));

    for (int i = 0; i < n; ++i)
        ones[i] = 1;

    return ones;
}

void matrixInverse(double* A, long int n) {
    long int *ipiv = (long int*)malloc((n+1)*sizeof(long int));
    long int lwork = n*n;
    double *work = (double*)malloc(lwork*sizeof(double));
    long int info;
    char C = 'N';
    long int nrhs = n;

    double *B = (double*)malloc(n*n*sizeof(double));

    for (int i = 0; i < n; ++i)
        B[i*n+i] = 1;

    dgetrf_(&n, &n, A, &n, ipiv, &info);
    dgetrs_(&C, &n, &nrhs, A, &n, ipiv, B, &n, &info);

    memcpy(A, B, n*n*sizeof(double));
    //dgetri_(&n, A, &n, ipiv, work, &lwork, &info);

    free(ipiv);
    free(work);
}

double vector_vector_mul(double* A, double* x, int n) {

    double y = cblas_ddot(n, A, 1, x, 1);

    return y;
}

double* matrix_vector_mul(double* A, double* x, int n) {

    int alpha = 1;
    int lda = n;
    int beta = 0;

    double *y = (double*)malloc(n*sizeof(double));

    // Ax + y;
    cblas_dgemv( 
        CblasColMajor, 
        CblasNoTrans,   
        n, n, alpha, 
        A, lda, 
        x, 1, 
        beta,
        y, 1 
    );

    
    return y;
}

double min_element(double* A, int n) {

    double result = INT_MAX;

    for (int i = 0; i < n; ++i)
        result = min(result, A[i]);

    return result;
}

double max_element(double *A, int n) {

    double result = -INT_MAX;

    for (int i = 0; i < n; ++i)
        result = max(result, A[i]);

    return result;
}

void computeEP(double* g_W0, double* g_M, 
               double* g_t1, double* g_t2,
               double g_a, double g_b,
               double g_alpha_l, double g_alpha_h,
               int g_m, double g_n) {

    for (int t_i = 0; t_i < g_m; ++t_i) {
        
        double t_alpha_a = g_alpha_l + g_a * (g_alpha_h - g_alpha_l);
        double t_alpha_b = g_alpha_l + g_b * (g_alpha_h - g_alpha_l);

        double t_delta = (t_alpha_b - t_alpha_a) / ((double)g_m - 1.0);
        double t_alpha = t_alpha_a + (t_delta * t_i);

        double t_lambda1 = (t_alpha * g_M[0]) + g_M[1];
        double t_lambda2 = (t_alpha * g_M[2]) + g_M[3];

        for (int j = 0; j < g_n; ++j) {
            int r = (t_i * g_n) + j;
            g_W0[r] = t_lambda1 * g_t1[j] + t_lambda2 * g_t2[j];
        }

    }
}

int main() {
    
    int m = 10000;
    int ASSETS = 1200;

    double g_a = 0.0;
    double g_b = 1.0;


    double* g_sigma = read_datasets("sigma4144x4144.csv", ASSETS, ASSETS);
    double* g_alpha = read_datasets("alpha4144.csv", ASSETS, 1);

    clock_t begin = clock();

    /* first phase */
    double *ones = initOnes(ASSETS);

    matrixInverse(g_sigma, ASSETS);

    double *g_t1 = matrix_vector_mul(g_sigma, g_alpha, ASSETS);
    double *g_t2 = matrix_vector_mul(g_sigma, ones, ASSETS);

    double g_M[4] = {0};

    g_M[0] = vector_vector_mul(g_alpha, g_t1, ASSETS);
    g_M[1] = g_M[2] = vector_vector_mul(g_alpha, g_t2, ASSETS);
    g_M[3] = vector_vector_mul(ones, g_t2, ASSETS);

    matrixInverse(g_M, 2);

    /* second phase */

    double g_alpha_l = min_element(g_alpha, ASSETS);
    double g_alpha_h = max_element(g_alpha, ASSETS);

    double *g_W0 = (double*)malloc(m*ASSETS*sizeof(double));

    computeEP(g_W0, g_M, g_t1, g_t2, g_a, g_b, g_alpha_l, g_alpha_h, m, ASSETS);

    clock_t end = clock();
    double elapsed_secs = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("[ASSETS = %d, m = %d] -> time = %.10lf\n", ASSETS, m, elapsed_secs);

    return 0;
}
