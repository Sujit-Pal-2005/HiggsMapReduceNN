import org.apache.spark.broadcast.Broadcast;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class NeuralNetwork {
    private NeuralNetwork() {}

    public static Network initWeights(int[] sizes, long seed) {
        Random rng = new Random(seed);
        int layers = sizes.length - 1;

        double[][][] weights = new double[layers][][];
        double[][] biases = new double[layers][];

        for (int l = 0; l < layers; l++) {
            int inputSize = sizes[l];
            int outputSize = sizes[l + 1];

            weights[l] = new double[outputSize][inputSize];
            biases[l] = new double[outputSize];
            double scale = Math.sqrt(inputSize);

            for (int j = 0; j < outputSize; j++) {
                for (int k = 0; k < inputSize; k++) {
                    weights[l][j][k] = rng.nextGaussian() / scale;
                }
            }
        }

        return new Network(weights, biases);
    }

    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static double[][] forwardPass(double[] x, Network model) {
        int layers = model.weights.length;
        double[][] activations = new double[layers + 1][];
        activations[0] = x;

        for (int l = 0; l < layers; l++) {
            double[] bias = model.biases[l];
            int outputSize = model.weights[l].length;
            activations[l + 1] = new double[outputSize];

            for (int j = 0; j < outputSize; j++) {
                double z = bias[j];
                for (int k = 0; k < activations[l].length; k++) {
                    z += model.weights[l][j][k] * activations[l][k];
                }
                activations[l + 1][j] = sigmoid(z);
            }
        }

        return activations;
    }

    public static List<Tuple2<Integer, Gradient>> mapFn(
            TrainingSamples record, Broadcast<Network> broadcastModel) {
        double[] x = record.x;
        double target = record.t;
        Network model = broadcastModel.value();
        double[][] activations = forwardPass(x, model);
        double[] output = activations[activations.length - 1];
        double[] delta = new double[output.length];

        for (int k = 0; k < output.length; k++) {
            delta[k] = (output[k] - target) * output[k] * (1.0 - output[k]);
        }

        List<Tuple2<Integer, Gradient>> emitted = new ArrayList<>();
        double[][][] weights = model.weights;

        for (int l = weights.length - 1; l >= 0; l--) {
            double[][] dW = new double[delta.length][activations[l].length];

            for (int j = 0; j < delta.length; j++) {
                for (int k = 0; k < activations[l].length; k++) {
                    dW[j][k] = delta[j] * activations[l][k];
                }
            }

            emitted.add(new Tuple2<>(l, new Gradient(dW, delta.clone())));

            if (l > 0) {
                double[] previousDelta = new double[activations[l].length];

                for (int j = 0; j < activations[l].length; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < delta.length; k++) {
                        sum += weights[l][k][j] * delta[k];
                    }
                    previousDelta[j] =
                            activations[l][j] * (1.0 - activations[l][j]) * sum;
                }
                delta = previousDelta;
            }
        }

        return emitted;
    }

    public static Gradient reduceFn(Gradient g1, Gradient g2) {
        double[][] dW = new double[g1.dW.length][];
        double[] db = new double[g1.db.length];

        for (int j = 0; j < g1.dW.length; j++) {
            dW[j] = new double[g1.dW[j].length];
            for (int k = 0; k < g1.dW[j].length; k++) {
                dW[j][k] = g1.dW[j][k] + g2.dW[j][k];
            }
        }

        for (int j = 0; j < g1.db.length; j++) {
            db[j] = g1.db[j] + g2.db[j];
        }

        return new Gradient(dW, db);
    }

    public static Tuple2<String, Long> predictCounts(
            TrainingSamples record, Broadcast<Network> broadcastModel) {
        Network model = broadcastModel.value();
        double[] output = forwardPass(record.x, model)[model.weights.length];

        int predicted = output[0] >= 0.5 ? 1 : 0;
        int actual = record.t == 1.0 ? 1 : 0;
        String key;

        if (predicted == 1 && actual == 1) {
            key = "TP";
        } else if (predicted == 1) {
            key = "FP";
        } else if (actual == 1) {
            key = "FN";
        } else {
            key = "TN";
        }

        return new Tuple2<>(key, 1L);
    }
}