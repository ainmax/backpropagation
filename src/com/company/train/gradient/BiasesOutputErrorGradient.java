package com.company.train.gradient;

import com.company.model.network.NeuralNetwork;
import com.company.train.TestSet;
import com.company.train.gradient.NetworkOutputErrorGradient;

public class BiasesOutputErrorGradient extends NetworkOutputErrorGradient {
    public BiasesOutputErrorGradient(NeuralNetwork network, TestSet.Test test) {
        super(network, test);
    }

    double calcLinearFirstDerivative(double linearCoefficient) {
        return 1;
    }

    int getConnectedParametersCount(int previousLayerSize, int currentLayerSize) {
        return currentLayerSize;
    }
}
