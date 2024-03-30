package com.company.train;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

public class TestSet {
    private final int inputSize;
    private final Function<double[], double[]> answerFunction;
    private final List<Test> tests = new ArrayList<>();
    public final int size;

    private int currentTestIndex = -1;

    public TestSet(int inputSize, Function<double[], double[]> answerFunction) {
        this.inputSize = inputSize;
        this.answerFunction = answerFunction;

        generateBitmaskTestSet();
        size = tests.size();
    }

    // Generates all bitmasks with length inputSize
    private void generateBitmaskTestSet() {
        double[] bitmask = new double[inputSize];

        for (int i = 0; i < (int)Math.pow(2, inputSize); ++i) {
            for (int j = 0; j < inputSize; ++j) {
                if (bitmask[j] == 0) {
                    bitmask[j] = 1;
                    break;
                }

                bitmask[j] = 0;
            }

            double[] clonedBitmask = bitmask.clone();
            tests.add(new Test(clonedBitmask, answerFunction.apply(clonedBitmask)));
        }

        Collections.shuffle(tests);
    }

    private void generateTwoPowersBitmasks() {
        double[] bitmask = new double[inputSize];
        int size = inputSize + 1;

        double[] clonedBitmask = bitmask.clone();
        tests.add(new Test(clonedBitmask, answerFunction.apply(clonedBitmask)));

        for (int i = 1; i < size; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                if (bitmask[j] == 0) {
                    bitmask[j] = 1;
                    break;
                }
            }

            clonedBitmask = bitmask.clone();
            tests.add(new Test(clonedBitmask, answerFunction.apply(clonedBitmask)));
        }

        Collections.shuffle(tests);
    }

    // Generates one simple test (1, 1, ..., 1)
    private void generateOneTest() {
        double[] bitmask = new double[inputSize];
        for (int j = 0; j < inputSize; ++j) {
            bitmask[j] = 1;
        }

        tests.add(new Test(bitmask, answerFunction.apply(bitmask)));
    }

    public Test nextTest() {
        if (!hasNextTest()) {
            throw new RuntimeException("There is no next test.");
        }

        return tests.get(++currentTestIndex);
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
