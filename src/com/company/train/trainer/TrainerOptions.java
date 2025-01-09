package com.company.train.trainer;

public record TrainerOptions(double learnSpeed, double inertiaCoefficient, int trainEpochsCount, int batchSize, double maxAcceptableAverageOutputError, double maxAcceptableOutputError) {
}
