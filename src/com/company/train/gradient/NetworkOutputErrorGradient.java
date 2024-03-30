package com.company.train.gradient;

import com.company.model.Matrix;
import com.company.model.network.NeuralNetwork;
import com.company.train.TestSet;

abstract class NetworkOutputErrorGradient {
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
           M - W(j) h-dimension,
           Gij[u][v] - partial derivative of A(i)[u] by W(j)[v]
     */

    public NetworkOutputErrorGradient(NeuralNetwork network, TestSet.Test test) {
        this.network = network;
        layersGradients = new Matrix[network.hiddenLayersCount + 1];

        Matrix input = new Matrix(network.inputSize, 1, test.input());
        calcNextLayerGradient(0, network.inputSize, input);

        errorFunctionGradient = calcErrorFunctionGradient(test);
    }

    public double[] getOutputErrorGradient() {
        return errorFunctionGradient;
    }

    // Sigmoid function first derivative
    private double calcActivatedNodeValueDerivative(double nodeCharge) {
        return 1 / (2 + Math.pow(Math.exp(1), -nodeCharge) + Math.pow(Math.exp(1), nodeCharge));
    }

    abstract double calcLinearFirstDerivative(double linearCoefficient);

    abstract int getConnectedParametersCount(int previousLayerSize, int currentLayerSize);

    // Recursive function calculates all partial derivatives of next layer activations by all previous parameters, using partial derivatives which were calculated earlier
    private void calcNextLayerGradient(int currentLayerIndex, int previousLayerSize, Matrix previousLayerOutput) {
        // Formula for layer output is W * A + B, where W - weights between current and next layers, B - biases
        Matrix currentLayerInput = network.weights[currentLayerIndex].multiply(previousLayerOutput).plus(network.biases[currentLayerIndex]);
        int currentLayerSize = currentLayerInput.N;

        // Define empty gradient
        layersGradients[currentLayerIndex] = new Matrix(currentLayerSize, (currentLayerIndex == 0 ? 0 : layersGradients[currentLayerIndex - 1].M) + getConnectedParametersCount(previousLayerSize, currentLayerSize));

        // Calculate piece of gradient with partial derivatives parameters from current layer (by weights and biases, connected to current layer)
        for (int i = 0; i < currentLayerSize; ++i) {
            for (int k = 0; k < getConnectedParametersCount(previousLayerSize, currentLayerSize) / currentLayerSize; ++k) {
                layersGradients[currentLayerIndex].values[i][i * getConnectedParametersCount(previousLayerSize, currentLayerSize) / currentLayerSize + k + (currentLayerIndex == 0 ? 0 : layersGradients[currentLayerIndex - 1].M)] = calcLinearFirstDerivative(previousLayerOutput.values[k][0]);
                // Adds activation function partial derivative factor
                layersGradients[currentLayerIndex].values[i][i * getConnectedParametersCount(previousLayerSize, currentLayerSize) / currentLayerSize + k + (currentLayerIndex == 0 ? 0 : layersGradients[currentLayerIndex - 1].M)] *= calcActivatedNodeValueDerivative(currentLayerInput.values[i][0]);
            }
        }

        // If currentLayer - first, there is no deep gradient
        if (currentLayerIndex == 0) {
            Matrix currentLayerOutput = Matrix.sigmoidOf(currentLayerInput);
            calcNextLayerGradient(currentLayerIndex + 1, currentLayerSize, currentLayerOutput);
            return;
        }

        // Calculate piece of gradient with partial derivatives by parameters from previous layers (by weights and biases, NOT connected to current layer)
        Matrix deepGradient = network.weights[currentLayerIndex].multiply(layersGradients[currentLayerIndex - 1]);

        for (int i = 0; i < currentLayerSize; ++i) {
            for (int j = 0; j < layersGradients[currentLayerIndex - 1].M; ++j) {
                layersGradients[currentLayerIndex].values[i][j] = deepGradient.values[i][j];
                // Adds activation function partial derivative factor
                layersGradients[currentLayerIndex].values[i][j] *= calcActivatedNodeValueDerivative(currentLayerInput.values[i][0]);
            }
        }

        // If this statement true, previous layer is output, so output gradient calculated and recursion must be stopped
        if (currentLayerIndex >= network.hiddenLayersCount) {
            return;
        }

        Matrix currentLayerOutput = Matrix.sigmoidOf(currentLayerInput);
        calcNextLayerGradient(currentLayerIndex + 1, currentLayerSize, currentLayerOutput);
    }

    private double[] calcErrorFunctionGradient(TestSet.Test test) {
        double[] networkOutput = network.calcOutputBy(test.input());
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
}
