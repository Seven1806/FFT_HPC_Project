#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define PI 3.14159265358979323846
#define FFT_SIZE 8

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

// Optimized OpenMP FFT using Cooley-Tukey
void fft_parallel(Complex *X, int N) {
    if (N <= 1) return;

    int half = N / 2;
    Complex *X_even = (Complex *)malloc(half * sizeof(Complex));
    Complex *X_odd = (Complex *)malloc(half * sizeof(Complex));

    for (int i = 0; i < half; i++) {
        X_even[i] = X[i * 2];
        X_odd[i] = X[i * 2 + 1];
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        fft_parallel(X_even, half);
        #pragma omp section
        fft_parallel(X_odd, half);
    }

    #pragma omp parallel for schedule(guided)
    for (int k = 0; k < half; k++) {
        double angle = -2.0 * PI * k / N;
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

int main() {
    Complex *X = (Complex *)malloc(FFT_SIZE * sizeof(Complex));

    for (int i = 0; i < FFT_SIZE; i++) {
        X[i].real = (i < FFT_SIZE / 2) ? 1.0 : 0.0;
        X[i].imag = 0.0;
    }

    print_fft(X, FFT_SIZE, "Sinal Original");

    double start = omp_get_wtime();
    fft_parallel(X, FFT_SIZE);
    double end = omp_get_wtime();

    print_fft(X, FFT_SIZE, "Resultado FFT Paralela (OpenMP)");

    printf("Tempo de execução (Paralela): %.6f segundos\n", end - start);

    free(X);
    return 0;
}
