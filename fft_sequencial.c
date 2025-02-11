#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846
#define FFT_SIZE 8

typedef struct {
    double real;
    double imag;
} Complex;

// Função para imprimir os valores do vetor
void print_fft(Complex *X, int N, const char *title) {
    printf("\n%s:\n", title);
    for (int i = 0; i < N; i++) {
        printf("X[%d] = %.5f + %.5fi\n", i, X[i].real, X[i].imag);
    }
}

// FFT Sequencial
void fft_sequencial(Complex *X, int N) {
    if (N <= 1) return;

    Complex *X_even = (Complex *)malloc(N / 2 * sizeof(Complex));
    Complex *X_odd = (Complex *)malloc(N / 2 * sizeof(Complex));

    for (int i = 0; i < N / 2; i++) {
        X_even[i] = X[i * 2];
        X_odd[i] = X[i * 2 + 1];
    }

    fft_sequencial(X_even, N / 2);
    fft_sequencial(X_odd, N / 2);

    for (int k = 0; k < N / 2; k++) {
        double angle = -2 * PI * k / N;
        Complex t = {cos(angle) * X_odd[k].real - sin(angle) * X_odd[k].imag,
                     sin(angle) * X_odd[k].real + cos(angle) * X_odd[k].imag};

        X[k].real = X_even[k].real + t.real;
        X[k].imag = X_even[k].imag + t.imag;
        X[k + N / 2].real = X_even[k].real - t.real;
        X[k + N / 2].imag = X_even[k].imag - t.imag;
    }

    free(X_even);
    free(X_odd);
}

int main() {
    Complex *X = (Complex *)malloc(FFT_SIZE * sizeof(Complex));

    // Criando sinal de teste
    for (int i = 0; i < FFT_SIZE; i++) {
        X[i].real = (i < FFT_SIZE / 2) ? 1.0 : 0.0;
        X[i].imag = 0.0;
    }

    print_fft(X, FFT_SIZE, "Sinal Original");

    clock_t start = clock();
    fft_sequencial(X, FFT_SIZE);
    clock_t end = clock();

    print_fft(X, FFT_SIZE, "Resultado FFT Sequencial");

    printf("Tempo de execução (Sequencial): %.6f segundos\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(X);
    return 0;
}
