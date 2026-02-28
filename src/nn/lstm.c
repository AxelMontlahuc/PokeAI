#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "lstm.h"

// Distribution normale (moyenne 0, écart-type 1) avec la méthode de Box-Muller
double rand_normal() {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-10) u1 = 1e-10; // Pas de log(0)
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Produit scalaire canonique de R^n
double dot_product(double* x, double* y, int n) {
    double res = 0.0;
    for (int i = 0; i < n; i++) {
        res += x[i] * y[i];
    }
    return res;
}

// Norme euclidienne associée au produit scalaire canonique
double norm(double* x, int n) {
    return sqrt(dot_product(x, x, n));
}

// Extraction d'une colonne d'une matrice
double* column(double** matrix, int j, int rows) {
    double* col = malloc(rows * sizeof(double));
    for (int i = 0; i < rows; i++) {
        col[i] = matrix[i][j];
    }
    return col;
}

// Initialisation orthogonale : on prend la décomposition QR d'une matrice aléatoire et on utilise Q
void orthogonal_init(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand_normal();
        }
    }
    
    // Gram-Schmidt
    for (int j=0; j<cols; j++) {
        // Orthogonalisation
        double* projection = malloc(rows * sizeof(double));
        for (int i=0; i<rows; i++) {
            projection[i] = 0.0;
        }

        double* column_j = column(matrix, j, rows);

        for (int k=0; k<j; k++) {
            double* column_k = column(matrix, k, rows);

            double dot = dot_product(column_j, column_k, rows);
            for (int i=0; i<rows; i++) {
                projection[i] += dot * matrix[i][k];
            }

            free(column_k);
        }

        for (int i=0; i<rows; i++) {
            matrix[i][j] -= projection[i];
        }

        free(projection);
        free(column_j);

        column_j = column(matrix, j, rows);

        // Normalisation
        double n = norm(column_j, rows);
        if (n > 1e-6) { // Pas de division par zéro
            for (int i=0; i<rows; i++) {
                matrix[i][j] /= n;
            }
        }

        free(column_j);
    }
}