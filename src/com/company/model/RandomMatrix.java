package com.company.model;

import java.util.Random;

public class RandomMatrix extends Matrix {
    RandomMatrix(int N, int M, double minValue, double maxValue) {
        super(N, M);
        Random random = new Random();

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                values[i][j] = minValue + random.nextDouble() * (maxValue - minValue);
            }
        }
    }

    public RandomMatrix(int N, int M) {
        this(N, M, -1, 1);
    }
}
