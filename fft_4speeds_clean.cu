#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#define PI 3.14159265358979323846

typedef struct {
    double real;
    double imag;
} Complex;

// Function to print FFT results (first 16 values)
void print_fft(Complex *X, int N, const char *title) {
    printf("\n%s (first 16 values):\n", title);
    for (int i = 0; i < (N < 16 ? N : 16); i++) {
        printf("X[%d] = %.5f + %.5fi\n", i, X[i].real, X[i].imag);
    }
}

// 🔹 **Sequential Cooley-Tukey FFT**
void fft_sequential(Complex *X, int N) {
    if (N <= 1) return;

    int half = N / 2;
    Complex *X_even = (Complex *)malloc(half * sizeof(Complex));
    Complex *X_odd = (Complex *)malloc(half * sizeof(Complex));

    for (int i = 0; i < half; i++) {
        X_even[i] = X[i * 2];
        X_odd[i] = X[i * 2 + 1];
    }

    fft_sequential(X_even, half);
    fft_sequential(X_odd, half);

    for (int k = 0; k < half; k++) {
        double angle = -2 * PI * k / N;
        Complex twiddle = {cos(angle), sin(angle)};
        Complex t;
        t.real = twiddle.real * X_odd[k].real - twiddle.imag * X_odd[k].imag;
        t.imag = twiddle.real * X_odd[k].imag + twiddle.imag * X_odd[k].real;

        X[k].real = X_even[k].real + t.real;
        X[k].imag = X_even[k].imag + t.imag;
        X[k + half].real = X_even[k].real - t.real;
        X[k + half].imag = X_even[k].imag - t.imag;
    }

     free(X_even);
         free(X_odd);
     }
     
     // 🔹 **Parallel OpenMP FFT**
     void fft_parallel(Complex *X, int N) {
         int logN = __builtin_ctz(N); // Fast log2(N)
         int step, i, j, k;
         
         for (step = 2; step <= N; step *= 2) {
             int half_step = step / 2;
             double angle = -2.0 * PI / step;
             
             #pragma omp parallel for schedule(guided)
             for (k = 0; k < half_step; k++) {
                 Complex twiddle = {cos(k * angle), sin(k * angle)};
                 for (i = k; i < N; i += step) {
                     j = i + half_step;
                     Complex t;
                     t.real = twiddle.real * X[j].real - twiddle.imag * X[j].imag;
                     t.imag = twiddle.real * X[j].imag + twiddle.imag * X[j].real;
     
                     X[j].real = X[i].real - t.real;
                     X[j].imag = X[i].imag - t.imag;
                     X[i].real += t.real;
                     X[i].imag += t.imag;
                 }
             }
         }
     }

     // 🔹 **CUDA Kernel for GPU FFT**
     __global__ void fft_cuda_kernel(Complex *X, int N, int step) {
         int tid = threadIdx.x + blockIdx.x * blockDim.x;
         int half_step = step / 2;
     
         if (tid < N / 2) {
             int index = (tid / half_step) * step + (tid % half_step);
             double angle = -2.0 * PI * (tid % half_step) / step;
             Complex twiddle = {cos(angle), sin(angle)};
     
             Complex even = X[index];
             Complex odd = X[index + half_step];
     
             X[index].real = even.real + (twiddle.real * odd.real - twiddle.imag * odd.imag);
             X[index].imag = even.imag + (twiddle.real * odd.imag + twiddle.imag * odd.real);
             X[index + half_step].real = even.real - (twiddle.real * odd.real - twiddle.imag * odd.imag);
             X[index + half_step].imag = even.imag - (twiddle.real * odd.imag + twiddle.imag * odd.real);
         }
     }

     // 🔹 **GPU Wrapper for CUDA FFT**
     void fft_cuda(Complex *X, int N) {
         Complex *d_X;
         cudaMalloc(&d_X, N * sizeof(Complex));
         cudaMemcpy(d_X, X, N * sizeof(Complex), cudaMemcpyHostToDevice);
     
         int threads_per_block = 256;
         int num_blocks = (N + threads_per_block - 1) / threads_per_block;
     
         for (int step = 2; step <= N; step *= 2) {
             fft_cuda_kernel<<<num_blocks, threads_per_block>>>(d_X, N, step);
             cudaDeviceSynchronize();
         }
     
         cudaMemcpy(X, d_X, N * sizeof(Complex), cudaMemcpyDeviceToHost);
         cudaFree(d_X);
     }

      int main() {
          int sizes[] = {4096, 65536, 1048576, 16777216}; // Selected sizes
          int num_sizes = 4;
      
          for (int test = 0; test < num_sizes; test++) {
              int FFT_SIZE = sizes[test];
              printf("\n===============================\n");
              printf("Running FFT for size: %d\n", FFT_SIZE);
              printf("===============================\n");
      
              Complex *X_seq = (Complex *)malloc(FFT_SIZE * sizeof(Complex));
              Complex *X_par = (Complex *)malloc(FFT_SIZE * sizeof(Complex));
              Complex *X_gpu = (Complex *)malloc(FFT_SIZE * sizeof(Complex));
      
              for (int i = 0; i < FFT_SIZE; i++) {
                  X_seq[i].real = X_par[i].real = X_gpu[i].real = (i < FFT_SIZE / 2) ? 1.0 : 0.0;
                  X_seq[i].imag = X_par[i].imag = X_gpu[i].imag = 0.0;
              }
      
              double start, end;

              // Sequential
                      start = omp_get_wtime();
                      fft_sequential(X_seq, FFT_SIZE);
                      end = omp_get_wtime();
                      printf("\nExecution Time (Sequential FFT): %.6f seconds\n", end - start);
              
                      // OpenMP Parallel
                      start = omp_get_wtime();
                      fft_parallel(X_par, FFT_SIZE);
                      end = omp_get_wtime();
                      printf("\nExecution Time (Parallel OpenMP FFT): %.6f seconds\n", end - start);
              
                      // GPU CUDA
                      start = omp_get_wtime();
                      fft_cuda(X_gpu, FFT_SIZE);
                      end = omp_get_wtime();
                      printf("\nExecution Time (CUDA GPU FFT): %.6f seconds\n", end - start);
              
                      free(X_seq);
                      free(X_par);
                      free(X_gpu);
                  }
              
                  return 0;
              }
        	                	                                        	                        
        	                	                                        
        	                
       
