package com.company.model;

import java.util.Random;

public class RandomMatrix extends Matrix {
    RandomMatrix(int N, int M, double limit) {
        super(N, M);
        Random random = new Random();

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                values[i][j] = -limit + random.nextDouble() * 2 * limit;
            }
        }
    }

    public RandomMatrix(int N, int M) {
        this(N, M, Math.sqrt(6.0 / (N + M)));
    }
}
