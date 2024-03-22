package com.company.model;

import com.company.train.TestsBase;

abstract class NetworkGradient {
    private final NeuralNetwork network;
    private final Matrix[] layersGradients;
    private final double[] errorFunctionGradient;

    /*
    Gradient storage structure:

              W(0)     W(1)     W(2)    ...     W(n)
      A(0)  G(0)(0)     -        -      ...      -
      A(1)  G(1)(0)  G(1)(1)     -      ...      -
      A(2)  G(2)(0)  G(2)(1)  G(1)(2)   ...      -
        .      .        .        .       .       .
        .      .        .        .       .       .
        .      .        .        .       .       .
      A(n)  G(n)(0)  G(n)(1)  G(n)(2)   ...   G(n)(n)

      Where n = hiddenLayersQuantity,
      A(i) - nodes activations vector,
      W(j) - weights vector (weights matrices transform to vectors by indexing element (k, r) as [k * M + r], where M - matrix v-dimension),
      G(i)(j) = Gij - gradient matrix:

                            W(j)[0]         W(j)[1]         W(j)[2]      ...      W(j)[N * M - 1]
             A(i)[0]       Gij[0][0]       Gij[0][1]       Gij[0][2]     ...     Gij[0][N * M - 1]
             A(i)[1]       Gij[1][0]       Gij[1][1]       Gij[1][2]     ...     Gij[1][N * M - 1]
             A(i)[2]       Gij[2][0]       Gij[2][1]       Gij[2][2]     ...     Gij[2][N * M - 1]
                .              .               .               .          .              .
                .              .               .               .          .              .
                .              .               .               .          .              .
           A(i)[S - 1]   Gij[S - 1][0]   Gij[S - 1][1]   Gij[S - 1][2]   ...   Gij[S - 1][N * M - 1]

           Where N - W(j) v-dimension,
           M - W(j) - h-dimension,
           Gij[u][v] - partial derivative of A(i)[u] by W(j)[v]
     */

    NetworkGradient(NeuralNetwork network, double[] inputValues, TestsBase.Test test) {
        this.network = network;
        layersGradients = new Matrix[network.hiddenLayersCount + 1];

        calcFirstHiddenLayerGradient(inputValues);
        errorFunctionGradient = calcErrorFunctionGradient(inputValues, test);
    }

    // Sigmoid function first derivative
    private double calcActivatedNodeValueDerivative(double nodeCharge) {
        return 1 / (2 + Math.pow(Math.exp(1), -nodeCharge) + Math.pow(Math.exp(1), nodeCharge));
    }

    public double[] getErrorFunctionGradient() {
        return errorFunctionGradient;
    }

    // Calculates base of recursion - partial derivatives of first hiddenLayer activations by parameters, connected to inputLayer
    private void calcFirstHiddenLayerGradient(double[] inputValues) {
        Matrix inputLayer = new Matrix(inputValues.length, 1, inputValues);
        Matrix inputLayerOutput = network.weights[0].multiply(inputLayer).plus(network.biases[0]);
        layersGradients[0] = new Matrix(inputLayerOutput.N, network.inputSize * inputLayerOutput.N);

        // Calculate piece of gradient with partial derivatives parameters from first hidden layer (by weights and biases, connected to current layer)
        for (int i = 0; i < inputLayerOutput.N; ++i) {
            for (int k = 0; k < network.inputSize; ++k) {
                layersGradients[0].values[i][i * network.inputSize + k] = calcLinearFirstDerivative(inputValues[k]);
                // Adds activation function partial derivative factor
                layersGradients[0].values[i][i * network.inputSize + k] *= calcActivatedNodeValueDerivative(inputLayerOutput.values[i][0]);
            }
        }

        calcNextLayerGradient(1, inputLayerOutput, network.hiddenLayers[0].calcOutputBy(inputLayerOutput));
    }

    // Recursive function calculates all partial derivatives of next layer activations by all previous parameters, using partial derivatives which were calculated earlier
    private void calcNextLayerGradient(int currentLayerIndex, Matrix previousLayerInput, Matrix previousLayerOutput) {
        // If this statement true, previous layer is output, so output gradient calculated and recursion must be stopped
        if (currentLayerIndex >= network.hiddenLayersCount) {
            return;
        }

        int previousLayerSize = previousLayerInput.N;
        int currentLayerSize = previousLayerOutput.N;

        // Define empty gradient
        layersGradients[currentLayerIndex] = new Matrix(previousLayerOutput.N, layersGradients[currentLayerIndex - 1].M + previousLayerSize * currentLayerSize);


        // Calculate piece of gradient with partial derivatives parameters from current layer (by weights and biases, connected to current layer)
        for (int i = 0; i < currentLayerSize; ++i) {
            for (int k = 0; k < previousLayerSize; ++k) {
                layersGradients[currentLayerIndex].values[i][i * previousLayerSize + k + layersGradients[currentLayerIndex - 1].M] = calcLinearFirstDerivative(previousLayerInput.values[k][0]);
                // Adds activation function partial derivative factor
                layersGradients[currentLayerIndex].values[i][i * previousLayerSize + k + layersGradients[currentLayerIndex - 1].M] *= calcActivatedNodeValueDerivative(previousLayerOutput.values[i][0]);
            }
        }

        // Calculate piece of gradient with partial derivatives by parameters from previous layers (by weights and biases, NOT connected to current layer)
        Matrix deepGradient = network.weights[currentLayerIndex].multiply(layersGradients[currentLayerIndex - 1]);

        for (int i = 0; i < currentLayerSize; ++i) {
            for (int j = 0; j < layersGradients[currentLayerIndex - 1].M; ++j) {
                layersGradients[currentLayerIndex].values[i][j] = deepGradient.values[i][j];
                // Adds activation function partial derivative factor
                layersGradients[currentLayerIndex].values[i][j] *= calcActivatedNodeValueDerivative(previousLayerOutput.values[i][0]);
            }
        }

        Matrix nextLayerInput = network.hiddenLayers[currentLayerIndex].calcOutputBy(previousLayerOutput);
        calcNextLayerGradient(currentLayerIndex + 1, previousLayerOutput, nextLayerInput);
    }

    private double[] calcErrorFunctionGradient(double[] inputValues, TestsBase.Test test) {
        double[] networkOutput = network.calcOutputBy(inputValues);
        Matrix errorFunctionGradientMatrix = new Matrix(layersGradients[layersGradients.length - 1]);

        // Every partial derivative is multiplied by respective error function derivative
        for (int j = 0; j < errorFunctionGradientMatrix.N; ++j) {
            for (int k = 0; k < errorFunctionGradientMatrix.M; k++) {
                errorFunctionGradientMatrix.values[j][k] *= 2 * (networkOutput[j] - test.correctOutput()[j]);
            }
        }

        // Partial derivatives of output layer sums to partial derivatives of error function
        double[] errorFunctionGradient = new double[errorFunctionGradientMatrix.M];

        for (int j = 0; j < errorFunctionGradientMatrix.M; ++j) {
            for (int k = 0; k < errorFunctionGradientMatrix.N; k++) {
                errorFunctionGradient[j] += errorFunctionGradientMatrix.values[k][j];
            }
        }

        return errorFunctionGradient;
    }

    abstract double calcLinearFirstDerivative(double lineCoefficient);
}
