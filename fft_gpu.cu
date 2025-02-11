#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>

#define FFT_SIZE 8

void print_fft(cufftComplex *X, int N, const char *title) {
    printf("\n%s:\n", title);
    for (int i = 0; i < N; i++) {
        printf("X[%d] = %.5f + %.5fi\n", i, X[i].x, X[i].y);
    }
}

int main() {
    cufftComplex h_X[FFT_SIZE];

    for (int i = 0; i < FFT_SIZE; i++) {
        h_X[i].x = (i < FFT_SIZE / 2) ? 1.0 : 0.0;
        h_X[i].y = 0.0;
    }

    print_fft(h_X, FFT_SIZE, "Sinal Original");

    cufftComplex *d_X;
    cudaMalloc((void**)&d_X, sizeof(cufftComplex) * FFT_SIZE);
    cudaMemcpy(d_X, h_X, sizeof(cufftComplex) * FFT_SIZE, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, FFT_SIZE, CUFFT_C2C, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cufftExecC2C(plan, d_X, d_X, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaMemcpy(h_X, d_X, sizeof(cufftComplex) * FFT_SIZE, cudaMemcpyDeviceToHost);
    cufftDestroy(plan);
    cudaFree(d_X);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    print_fft(h_X, FFT_SIZE, "Resultado FFT na GPU");
    printf("Tempo de execução (GPU): %.6f milissegundos\n", milliseconds);

    return 0;
}
