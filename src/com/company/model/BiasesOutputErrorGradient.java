package com.company.model;

import com.company.train.TestsBase;

public class BiasesOutputErrorGradient extends NetworkOutputErrorGradient {
    public BiasesOutputErrorGradient(NeuralNetwork network, TestsBase.Test test) {
        super(network, test);
    }

    double calcLinearFirstDerivative(double linearCoefficient) {
        return 1;
    }

    int getConnectedParametersCount(int previousLayerSize, int currentLayerSize) {
        return currentLayerSize;
    }
}
