package com.company.model;

import java.util.Arrays;
import java.util.stream.IntStream;

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

    public Matrix(double[][] values) {
        if (values.length == 0) {
            throw new IllegalArgumentException("Bad argument for matrix initialization. There are no values.");
        }

        N = values.length;
        M = values[0].length;
        this.values = values;
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

    // Throws exception if dimensions don't equal
    public Matrix add(Matrix term) {
        if (N != term.N || M != term.M) {
            throw new IllegalArgumentException("Bad argument for matrix multiplication. Dimensions don't equal.");
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                values[i][j] += term.values[i][j];
            }
        }

        return this;
    }

    // Throws exception if dimensions don't equal
    public Matrix subtract(Matrix term) {
        if (N != term.N || M != term.M) {
            throw new IllegalArgumentException("Bad argument for matrix multiplication. Dimensions don't equal.");
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                values[i][j] -= term.values[i][j];
            }
        }

        return this;
    }

    // This - left operand, parameter - right operand
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

    // Each element in input transforms by sigmoid function
    public static Matrix sigmoidOf(Matrix input) {
        Matrix output = new Matrix(input);

        for (int i = 0; i < input.N; ++i) {
            output.values[i][0] = sigmoid(input.values[i][0]);
        }

        return output;
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.pow(Math.exp(1), -x));
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                result.append(values[i][j]);

                if (j + 1 < M) {
                    result.append(", ");
                }
            }

            result.append('\n');
        }

        return result.toString();
    }
}
