package com.company.model;

import com.company.train.TestsBase;

public class BiasesGradient extends NetworkGradient {
    public BiasesGradient(NeuralNetwork network, TestsBase.Test test) {
        super(network, test);
    }

    double calcLinearFirstDerivative(double lineCoefficient) {
        return 1;
    }
}
