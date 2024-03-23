package com.company.model;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

class MatrixTest {

    @Test
    void testMatrixAddition() {
        Matrix m1 = new Matrix(3, 4, new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        Matrix m2 = new Matrix(3, 4, new double[] {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});

        assert Arrays.deepEquals(m1.plus(m2).values, new Matrix(3, 4, new double[] {13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13}).values);
    }

    @Test
    void testMatrixMultiplication() {
        Matrix m1 = new Matrix(2, 3, new double[] {1, 2, 3, 4, 5, 6});
        Matrix m2 = new Matrix(3, 4, new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

        assert Arrays.deepEquals(m1.multiply(m2).values, new Matrix(2, 4, new double[] {38, 44, 50, 56, 83, 98, 113, 128}).values);
    }
}