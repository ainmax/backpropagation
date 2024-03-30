package com.company.train.trainer;

public record TrainerOptions(double learnSpeed, double inertiaCoefficient, int trainEpochsCount, double maxAcceptableAverageOutputError, double maxAcceptableOutputError) {
}
