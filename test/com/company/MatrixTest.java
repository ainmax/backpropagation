package com.company;

import com.company.model.Matrix;
import org.junit.jupiter.api.Test;

class MatrixTest {

    @Test
    void matrixInitializationException() {
        try {
            new Matrix(2, 3, new double[] {1.2, 2});
            assert false;
        } catch (IllegalArgumentException ignored) {

        }
    }

    @Test
    void matrixOperatorException() {
        try {
            Matrix matrix = new Matrix(2, 3, new double[] {1, 2, 3, 4, 5, 6});
            matrix.plus(new Matrix(2, 3, new double[] {2, 3, 4, 5, 6, 7}));
            matrix.multiply(new Matrix(3, 3, new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0}));
        } catch (IllegalArgumentException e) {
            assert false;
        }
    }
}