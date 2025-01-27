package com.company.model;

import com.company.model.network.NeuralNetwork;
import com.company.train.TestSet;
import com.company.train.trainer.Trainer;
import com.company.train.gradient.BiasesOutputErrorGradient;
import com.company.train.gradient.WeightsOutputErrorGradient;
import org.junit.jupiter.api.Test;

class WeightsGradientTest {
    @Test
    void testGradientCalculating() {
        NeuralNetwork network = new NeuralNetwork(3, 2, new int[] {2, 4});
        network.weights = new Matrix[] {new Matrix(2, 3, new double[] {-1, 0, 1, -2, 0, 2}), new Matrix(4, 2, new double[] {0, 1, 1, 0, 1, 1, 0, 0}), new Matrix(2, 4, new double[] {-2, -1, 1, 2, -4, -3, 3, 4})};
        network.biases = new Matrix[] {new Matrix(2, 1, new double[] {1, -1}), new Matrix(4, 1, new double[] {-1, 0, 0, 1}), new Matrix(2, 1, new double[] {1, -1})};

        WeightsOutputErrorGradient weightsOutputErrorGradient = new WeightsOutputErrorGradient(network, new TestSet.Test(new double[] {1, 0, 1}, new double[] {0.866, 0.688}));
        BiasesOutputErrorGradient biasesOutputErrorGradient = new BiasesOutputErrorGradient(network, new TestSet.Test(new double[] {1, 0, 1}, new double[] {0.866, 0.688}));

        assert Trainer.calcOutputError(weightsOutputErrorGradient.getOutputErrorGradient().values[0], new double[30]) < 0.000001;
        assert Trainer.calcOutputError(biasesOutputErrorGradient.getOutputErrorGradient().values[0], new double[30]) < 0.000001;
    }
}