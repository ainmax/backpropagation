package com.company.train;

import java.util.function.Function;

public class TestsBase {
    private final int inputSize;
    private final Function<double[], double[]> answerFunction;
    private Test[] tests;
    final int size;

    private int currentTestIndex = -1;

    public TestsBase(int inputSize, Function<double[], double[]> answerFunction) {
        this.inputSize = inputSize;
        this.answerFunction = answerFunction;

        generateBitmaskTestsBase();
        size = tests.length;
    }

    // Generates all bitmasks with length inputSize
    private void generateBitmaskTestsBase() {
        tests = new Test[(int)Math.pow(2, inputSize)];
        double[] bitmask = new double[inputSize];

        for (int i = 0; i < tests.length; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                if (bitmask[j] == 0) {
                    bitmask[j] = 1;
                    break;
                }

                bitmask[j] = 0;
            }

            double[] clonedBitmask = bitmask.clone();
            tests[i] = new Test(clonedBitmask, answerFunction.apply(clonedBitmask));
        }
    }

    Test nextTest() {
        if (!hasNextTest()) {
            throw new RuntimeException("There is no next test.");
        }

        return tests[++currentTestIndex];
    }

    void clearTestsQueue() {
        currentTestIndex = -1;
    }

    boolean hasNextTest() {
        return currentTestIndex + 1 < size;
    }

    public record Test(double[] input, double[] correctOutput) {}
}
