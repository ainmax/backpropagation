package com.company.model;

import com.company.train.TestsBase;

public class WeightsGradient extends NetworkGradient {
    public WeightsGradient(NeuralNetwork network, TestsBase.Test test) {
        super(network, test);
    }

    double calcLinearFirstDerivative(double lineCoefficient) {
        return lineCoefficient;
    }
}
