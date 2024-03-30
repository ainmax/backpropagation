package com.company.train.gradient;

import com.company.model.network.NeuralNetwork;
import com.company.train.TestSet;
import com.company.train.gradient.NetworkOutputErrorGradient;

public class WeightsOutputErrorGradient extends NetworkOutputErrorGradient {
    public WeightsOutputErrorGradient(NeuralNetwork network, TestSet.Test test) {
        super(network, test);
    }

    double calcLinearFirstDerivative(double linearCoefficient) {
        return linearCoefficient;
    }

    int getConnectedParametersCount(int previousLayerSize, int currentLayerSize) {
        return previousLayerSize * currentLayerSize;
    }
}
