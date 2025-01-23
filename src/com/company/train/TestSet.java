package com.company.train;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;
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

        generateDigitsImagesTestSet();
//        generateBitmaskTestSet();

        size = tests.size();
    }

    private void generateDigitsImagesTestSet() {
        FileInputStream inputStream = null;
        Scanner sc = null;

        try {
            inputStream = new FileInputStream("C:/Users/aynur/desktop/nn/dataset/MNIST_txt/MNIST_train.txt");
            sc = new Scanner(inputStream, StandardCharsets.UTF_8);

            String[] pixels;
            int answer;

            double[] correctOutput;
            double[] input;

            while (sc.hasNextLine()) {
                String line = sc.nextLine();

                pixels = Arrays.copyOfRange(line.split(","), 1, line.split(",").length);
                answer = Integer.parseInt(line.substring(0, 1));

                input = new double[pixels.length];
                correctOutput = new double[10];

                for (int i = 0; i < pixels.length; ++i) {
                    input[i] = Integer.parseInt(pixels[i]) / 255.0;
                }

                correctOutput[answer] = 1.0;


                tests.add(new Test(input.clone(), correctOutput.clone()));
            }

            if (sc.ioException() != null) {
                throw sc.ioException();
            }
        } catch(Exception e) {
            System.out.println(e.getMessage());
        }

        if (inputStream != null) {
            try {
                inputStream.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        if (sc != null) {
            sc.close();
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

    // Generates bitmasks like (1, 1, ..., 1, 0, ..., 0)
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
