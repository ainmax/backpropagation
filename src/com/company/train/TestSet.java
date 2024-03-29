package com.company.train;

import java.util.function.Function;

public class TestSet {
    private final int inputSize;
    private final Function<double[], double[]> answerFunction;
    private Test[] tests;
    public final int size;

    private int currentTestIndex = -1;

    public TestSet(int inputSize, Function<double[], double[]> answerFunction) {
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

    // Generates one simple test (1, 1, ..., 1)
    private void generateOneTest() {
        tests = new Test[1];

        double[] bitmask = new double[inputSize];
        for (int j = 0; j < inputSize; ++j) {
            bitmask[j] = 1;
        }

        tests[0] = new Test(bitmask, answerFunction.apply(bitmask));
    }

    public Test nextTest() {
        if (!hasNextTest()) {
            throw new RuntimeException("There is no next test.");
        }

        return tests[++currentTestIndex];
    }

    public void clearTestsQueue() {
        currentTestIndex = -1;
    }

    public boolean hasNextTest() {
        return currentTestIndex + 1 < size;
    }

    public int getCurrentTestIndex() {
        return currentTestIndex;
    }

    public record Test(double[] input, double[] correctOutput) {}
}
