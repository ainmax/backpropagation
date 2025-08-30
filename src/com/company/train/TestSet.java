package com.company.train;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Stream;

public class TestSet {
    private final int inputSize;
    private final Function<double[], double[]> answerFunction;
    private final List<Test> tests = new ArrayList<>();
    public final int size;

    private int currentTestIndex = -1;

    public TestSet(int inputSize, Function<double[], double[]> answerFunction) {
        this.inputSize = inputSize;
        this.answerFunction = answerFunction;

        generateDigitsImagesTestSet("C:/Users/Айнур/Desktop/traindataset7819.txt");
//        generateBitmaskTestSet();

        size = tests.size();
    }

    private void generateDigitsImagesTestSet(String path) {
        ArrayList<String> lines = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String nextLine = reader.readLine();
            while (nextLine != null) {
                lines.add(nextLine);
                nextLine = reader.readLine();
            }
        } catch (IOException e) {
            System.out.println("Something wrong with dataset reading");
        }

        for (int i = 0; i < lines.size(); ++i) {
            double[] testOutput = new double[10];
            testOutput[Integer.parseInt(lines.get(i).substring(0, 1))] = 1;

            Stream<Double> pixelValuesStream = Arrays.stream(lines.get(i).split(",")).map(number -> Double.parseDouble(number) / 255);
            ArrayList<Double> pixelValues = new ArrayList<>(pixelValuesStream.toList());

            double[] testInput = new double[pixelValues.size() - 1];
            for (int j = 0; j < pixelValues.size() - 1; ++j) {
                testInput[j] = pixelValues.get(j + 1);
            }

            tests.add(new Test(testInput, testOutput));
        }

        Collections.shuffle(tests);
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

    // Generates all bitmasks with length inputSize
    private void generateDefectiveBitmaskTestSet() {
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

            if (!Arrays.equals(clonedBitmask, new double[] {1, 1, 0, 0, 0, 1, 1, 0, 1, 0})) {
                tests.add(new Test(clonedBitmask, answerFunction.apply(clonedBitmask)));
            }
        }

        Collections.shuffle(tests);
    }

    // Generates bitmasks like (1, ..., 1, 0, ..., 0)
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
        Collections.shuffle(tests);
    }

    public boolean hasNextTest() {
        return currentTestIndex + 1 < size;
    }

    public int getCurrentTestIndex() {
        return currentTestIndex;
    }

    public record Test(double[] input, double[] correctOutput) {}
}
