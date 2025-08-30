package com.company.model.network;

import com.company.model.Matrix;
import com.company.model.RandomMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.stream.Stream;

// Neural network model
public class NeuralNetwork {
    // Options
    public final int inputSize;
    public final int outputSize;
    public final int hiddenLayersCount;
    public final int[] hiddenLayersSizes;

    // Current parameters values
    public Matrix[] weights;
    public Matrix[] biases;

    // Throws exception if hiddenLayers is empty
    public NeuralNetwork(int inputSize, int outputSize, int[] hiddenLayersSizes) {
        hiddenLayersCount = hiddenLayersSizes.length;

        if (hiddenLayersCount == 0) {
            throw new IllegalArgumentException("There is no way to create network without hidden layers.");
        }

        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.hiddenLayersSizes = hiddenLayersSizes.clone();

        fillParametersWithRandomValues();
    }

    // Creates network by parameters which given in console
    public NeuralNetwork(int inputSize, int outputSize, int[] hiddenLayersSizes, String separator) {
        hiddenLayersCount = hiddenLayersSizes.length;

        if (hiddenLayersCount == 0) {
            throw new IllegalArgumentException("There is no way to create network without hidden layers.");
        }

        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.hiddenLayersSizes = hiddenLayersSizes.clone();

        fillParametersWithRandomValues();

        Scanner in = new Scanner(System.in);

        for (int i = 0; i < hiddenLayersCount + 1; ++i) {
            ArrayList<double[]> matrix = new ArrayList<>();

            while (true) {
                String[] line = in.nextLine().split(", ");

                if (line[0].equals(separator)) {
                    break;
                }

                double[] values = new double[line.length];

                for (int j = 0; j < line.length; ++j) {
                    values[j] = Double.parseDouble(line[j]);
                }

                matrix.add(values);
            }

            for (int j = 0; j < weights[i].N; ++j) {
                for (int k = 0; k < weights[i].M; ++k) {
                    weights[i].values[j][k] = matrix.get(j)[k];
                }
            }

            matrix = new ArrayList<>();

            while (true) {
                String[] line = in.nextLine().split(", ");

                if (line[0].equals(separator)) {
                    break;
                }

                double[] values = new double[line.length];

                for (int j = 0; j < line.length; ++j) {
                    values[j] = Double.parseDouble(line[j]);
                }

                matrix.add(values);
            }

            for (int j = 0; j < biases[i].N; ++j) {
                for (int k = 0; k < biases[i].M; ++k) {
                    biases[i].values[j][k] = matrix.get(j)[k];
                }
            }
        }
    }

    // Creates network by another network's options and parameters
    public NeuralNetwork(NeuralNetwork network) {
        inputSize = network.inputSize;
        outputSize = network.outputSize;
        hiddenLayersCount = network.hiddenLayersCount;
        hiddenLayersSizes = network.hiddenLayersSizes.clone();

        fillParametersWithRandomValues();

        for (int i = 0; i < weights.length; ++i) {
            weights[i] = new Matrix(network.weights[i]);
        }

        for (int i = 0; i < biases.length; ++i) {
            biases[i] = new Matrix(network.biases[i]);
        }
    }

    private void fillParametersWithRandomValues() {
        biases = new Matrix[hiddenLayersCount + 1];
        weights = new Matrix[hiddenLayersCount + 1];

        // Fill parameters
        weights[0] = new RandomMatrix(hiddenLayersSizes[0], inputSize);

        for (int i = 0; i < hiddenLayersCount; ++i) {
            biases[i] = new Matrix(hiddenLayersSizes[i], 1);

            if (i < hiddenLayersCount - 1) {
                weights[i + 1] = new RandomMatrix(hiddenLayersSizes[i + 1], hiddenLayersSizes[i]);
            }
        }

        biases[hiddenLayersCount] = new Matrix(outputSize, 1);
        weights[hiddenLayersCount] = new RandomMatrix(outputSize, hiddenLayersSizes[hiddenLayersCount - 1]);
    }

    public double[] calcOutputBy(double[] inputValues) {
        Matrix previousLayerOutput = new Matrix(inputSize, 1, inputValues);

        for (int i = 0; i < hiddenLayersCount; ++i) {
            // Formula for layer output is W * A + B, where W - weights between current and next layers, B - biases
            previousLayerOutput = Matrix.sigmoidOf(weights[i].multiply(previousLayerOutput).plus(biases[i]));
        }

        // Formula for layer output is W * A + B, where W - weights between current and next layers, B - biases
        Matrix outputMatrix = Matrix.sigmoidOf(weights[hiddenLayersCount].multiply(previousLayerOutput).plus(biases[hiddenLayersCount]));

        // Convert matrix to array
        double[] output = new double[outputSize];

        for (int i = 0; i < outputSize; ++i) {
            output[i] = outputMatrix.values[i][0];
        }

        return output;
    }
}
