package com.company.train.trainer;

import com.company.model.Matrix;
import com.company.model.network.NeuralNetwork;
import com.company.train.TestSet;
import com.company.train.gradient.BiasesOutputErrorGradient;
import com.company.train.gradient.WeightsOutputErrorGradient;

import java.util.ArrayList;
import java.util.concurrent.*;

public class OfflineTrainer extends Trainer {
    public OfflineTrainer(NeuralNetwork network, TestSet testSet, TrainerOptions options) {
        super(network, testSet, options);
    }

    NeuralNetwork trainEpoch(NeuralNetwork network) {
        // Define zero increments
        previousWeightsIncrements = new Matrix[network.weights.length];
        previousBiasesIncrements = new Matrix[network.biases.length];

        for (int i = 0; i < network.weights.length; ++i) {
            previousWeightsIncrements[i] = new Matrix(network.weights[i].N, network.weights[i].M);
        }

        for (int i = 0; i < network.biases.length; ++i) {
            previousBiasesIncrements[i] = new Matrix(network.biases[i].N, network.biases[i].M);
        }

        testSet.clearTestsQueue();

        for (int batchIndex = 0; testSet.hasNextTest(); ++batchIndex) {
            try {
                trainBatch(network, batchIndex);
            } catch(Exception e) {
                System.out.println(e.getMessage());
            }
        }

        return network;
    }

    private void trainBatch(NeuralNetwork network, int batchIndex) throws ExecutionException, InterruptedException {
        int currentBatchSize = Math.min(options.batchSize(), testSet.size - testSet.getCurrentTestIndex() - 1);

        // Save old parameters
        Matrix[] oldWeights = new Matrix[network.weights.length];
        Matrix[] oldBiases = new Matrix[network.biases.length];

        for (int p = 0; p < network.weights.length; ++p) {
            oldWeights[p] = new Matrix(network.weights[p]);
        }

        for (int p = 0; p < network.biases.length; ++p) {
            oldBiases[p] = new Matrix(network.biases[p]);
        }

        // generate batch
        TestSet.Test[] batchTests = new TestSet.Test[currentBatchSize];
        for (int i = 0; testSet.hasNextTest() && testSet.getCurrentTestIndex() < (batchIndex + 1) * options.batchSize() - 1; ++i) {
            batchTests[i] = testSet.nextTest();
        }

        // initialize processes
        final int PROCESSES_COUNT = 8;

        ExecutorService executor = Executors.newFixedThreadPool(PROCESSES_COUNT);
        ArrayList<Future<Matrix[]>> futures = new ArrayList<>();
        Callable<Matrix[]>[] processes = new Callable[PROCESSES_COUNT];

        for (int processIndex = 0; processIndex < PROCESSES_COUNT; ++processIndex) {
            int firstTestIndex = processIndex * currentBatchSize / PROCESSES_COUNT;
            int lastTestIndex = (processIndex + 1) * currentBatchSize / PROCESSES_COUNT;

            processes[processIndex] = new GradientCalculation(batchTests, firstTestIndex, lastTestIndex, network);
        }

        // start processes
        for (int processIndex = 0; processIndex < PROCESSES_COUNT; ++processIndex) {
            Future<Matrix[]> future = executor.submit(processes[processIndex]);
            futures.add(future);
        }

        // sum gradients
        int processIndex = 0;

        Matrix weightsGradient = futures.getFirst().get()[0];
        Matrix biasesGradient = futures.getFirst().get()[1];

        ++processIndex;

        for (; processIndex < PROCESSES_COUNT; ++processIndex) {
            weightsGradient.add(futures.get(processIndex).get()[0]);
            biasesGradient.add(futures.get(processIndex).get()[1]);
        }

        executor.shutdown();

        // Calculating average gradients by all tests
        for (int i = 0; i < weightsGradient.M; ++i) {
            weightsGradient.values[0][i] /= currentBatchSize;
        }

        for (int i = 0; i < biasesGradient.M; ++i) {
            biasesGradient.values[0][i] /= currentBatchSize;
        }

        // Tweak network's parameters
        network = tweakNetworkParametersByGradients(network, weightsGradient.values[0], biasesGradient.values[0]);

        // Calculate new increments of weights
        for (int p = 0; p < network.weights.length; ++p) {
            previousWeightsIncrements[p] = network.weights[p].subtract(oldWeights[p]);
        }

        // Calculate new increments of biases
        for (int p = 0; p < network.biases.length; ++p) {
            previousBiasesIncrements[p] = network.biases[p].subtract(oldBiases[p]);
        }
    }

    static class GradientCalculation implements Callable<Matrix[]> {
        TestSet.Test[] tests;
        final int firstTestIndex;
        final int lastTestIndex;
        NeuralNetwork network;

        GradientCalculation(TestSet.Test[] tests, int firstTestIndex, int lastTestIndex, NeuralNetwork network) {
            this.tests = tests;
            this.firstTestIndex = firstTestIndex;
            this.lastTestIndex = lastTestIndex;
            this.network = network;
        }

        @Override
        public Matrix[] call() {
            Matrix[] gradients = new Matrix[2];

            if (firstTestIndex == lastTestIndex) {
                return gradients;
            }

            int testIndex = firstTestIndex;

            gradients[0] = new WeightsOutputErrorGradient(network, tests[testIndex]).getOutputErrorGradient();
            gradients[1] = new BiasesOutputErrorGradient(network, tests[testIndex]).getOutputErrorGradient();

            ++testIndex;

            for (; testIndex < lastTestIndex; ++testIndex) {
                gradients[0] = gradients[0].plus(new WeightsOutputErrorGradient(network, tests[testIndex]).getOutputErrorGradient());
                gradients[1] = gradients[1].plus(new BiasesOutputErrorGradient(network, tests[testIndex]).getOutputErrorGradient());
            }

            return gradients;
        }
    }
}
