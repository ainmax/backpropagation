package com.company.model;

import java.util.Arrays;

public class Matrix {
    public final int N; // Vertical (v-) dimension (first index)
    public final int M; // Horizontal (h-) dimension (second index)

    public final double[][] values;

    // Makes matrix by dimensions and vector of values. Values divide to N groups by M.
    // Throws exception if dimensions don't match to values quantity (v-dimension and h-dimension multiplication)
    public Matrix(int N, int M, double[] values) {
        if (values.length != N * M) {
            throw new IllegalArgumentException("Bad argument for matrix initialization. Values quantities don't match dimensions.");
        }

        this.N = N;
        this.M = M;
        this.values = new double[N][M];

        for (int i = 0; i < N; ++i) {
            System.arraycopy(values, M * i, this.values[i], 0, M);
        }
    }

    // Makes matrix by another matrix
    public Matrix(Matrix matrix) {
        N = matrix.N;
        M = matrix.M;
        values = new double[this.N][];

        for (int i = 0; i < N; ++i) {
            values[i] = matrix.values[i].clone();
        }
    }

    // Makes empty matrix
    public Matrix(int N, int M) {
        this.N = N;
        this.M = M;
        values = new double[N][M];
    }

    // Throws exception if dimensions don't equal
    public Matrix plus(Matrix term) {
        if (N != term.N || M != term.M) {
            throw new IllegalArgumentException("Bad argument for matrix multiplication. Dimensions don't equal.");
        }

        Matrix sum = new Matrix(this);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                sum.values[i][j] += term.values[i][j];
            }
        }

        return sum;
    }

    // This - first operand, parameter - second operand
    // Throws exception if dimensions don't match criteria of matrix multiplication
    public Matrix multiply(Matrix factor) {
        if (M != factor.N) {
            throw new IllegalArgumentException("Bad argument for matrix multiplication. Dimensions don't match criteria.");
        }

        Matrix multiplication = new Matrix(N, factor.M);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < factor.M; ++j) {
                for (int k = 0; k < M; ++k) {
                    multiplication.values[i][j] += values[i][k] * factor.values[k][j];
                }
            }
        }

        return multiplication;
    }

    public Matrix multiply(double factor) {
        Matrix multiplication = new Matrix(N, M);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                multiplication.values[i][j] = values[i][j] * factor;
            }
        }

        return multiplication;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder(Arrays.toString(values[0]));

        for (int i = 1; i < N; ++i) {
            result.append('\n').append(Arrays.toString(values[i]));
        }

        return result.toString();
    }
}
