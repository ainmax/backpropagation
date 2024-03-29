package com.company.train;

public record TrainerOptions(double learnSpeed, double inertiaCoefficient, int trainEpochsCount, double maxAcceptableAverageOutputError, double maxAcceptableOutputError) {}
