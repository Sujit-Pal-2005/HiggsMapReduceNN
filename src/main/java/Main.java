
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.Column;

import java.io.Serializable;
import java.util.Arrays;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.stddev;
import static org.apache.spark.sql.functions.col;
import org.apache.spark.broadcast.Broadcast;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Main {

    static final String DATA_PATH = "data/HIGGS.csv";
    static final int MAX_ROWS = 1_000_000;
    static final double TEST_FRACTION = 0.2;
    static final int[] HIDDEN_LAYERS = {16, 8};
    static final int EPOCHS = 20;
    static final int BATCH_SIZE = 10_000;
    static final double ALPHA = 0.5;
    static final long SEED = 42L;
    static final int N_FEATURES = 28;

    public static void main(String[] args) {
        String path = args.length > 0 ? args[0] : DATA_PATH;

        // ---- initialize Spark ----

        SparkSession spark = SparkSession.builder()
                .appName("HiggsMapReduceNN")
                .master("local[*]")
                .config("spark.local.dir", System.getProperty("user.home") + "/spark-tmp")
                .getOrCreate();


        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

        spark.sparkContext().setLogLevel("WARN");

        // ---- load data ----
        StructType schema = new StructType()
                .add("label", DataTypes.DoubleType);

        for (int i = 0; i < N_FEATURES; i++) {
            schema = schema.add("f" + i, DataTypes.DoubleType);
        }

        Dataset<Row> df = spark.read()
                .schema(schema)
                .option("header", "false")
                .csv(path);

        if (MAX_ROWS > 0) {
            df = df.limit(MAX_ROWS) ;
        }

        // ---- train / test split ----
        Dataset<Row>[] split = df.randomSplit(
                new double[]{1.0 - TEST_FRACTION, TEST_FRACTION},
                SEED
        );

        Dataset<Row> trainDf = split[0];
        Dataset<Row> testDf = split[1];

        // ---- standardize using training statistics ----
        String[] features = new String[N_FEATURES];

        for (int i = 0; i < N_FEATURES; i++) {
            features[i] = "f" + i;
        }

        Column[] aggregateColumns = new Column[N_FEATURES * 2];

        for (int i = 0; i < N_FEATURES; i++) {
            aggregateColumns[i] =
                    avg(col(features[i])).alias("m_" + features[i]);

            aggregateColumns[N_FEATURES + i] =
                    stddev(col(features[i])).alias("s_" + features[i]);
        }

        Row stats = trainDf
                .agg(
                        aggregateColumns[0],
                        Arrays.copyOfRange(
                                aggregateColumns, 1, aggregateColumns.length
                        )
                )
                .first();

        double[] mean = new double[N_FEATURES];
        double[] std = new double[N_FEATURES];

        for (int i = 0; i < N_FEATURES; i++) {
            mean[i] = stats.getDouble(i);

            double standardDeviation =
                    stats.isNullAt(N_FEATURES + i)
                            ? 0.0
                            : stats.getDouble(N_FEATURES + i);

            std[i] = standardDeviation + 1e-8;
        }

        // ---- convert rows into training samples ----
        JavaRDD<TrainingSamples> trainRdd = trainDf.javaRDD()
                .map(row -> toTrainingSamples(row, mean, std))
                .cache();

        JavaRDD<TrainingSamples> testRdd = testDf.javaRDD()
                .map(row -> toTrainingSamples(row, mean, std))
                .cache();

        long nTrain = trainRdd.count();
        long nTest = testRdd.count();

        System.out.printf(
                "train rows: %d   test rows: %d%n",
                nTrain, nTest
        );

        // ---- create mini-batches ----
        int nBatches = Math.max(
                1,
                (int) Math.ceil(nTrain / (double) BATCH_SIZE)
        );

        double[] weights = new double[nBatches];
        Arrays.fill(weights, 1.0);

        JavaRDD<TrainingSamples>[] batches =
                trainRdd.randomSplit(weights, SEED);

        long[] batchSizes = new long[nBatches];

        for (int i = 0; i < nBatches; i++) {
            batches[i].cache();
            batchSizes[i] = batches[i].count();
        }

        System.out.printf(
                "mini-batches: %d (about %d rows each) "
                        + "-> %d weight updates per epoch%n",
                nBatches, BATCH_SIZE, nBatches
        );

        // ---- network architecture ----
        int[] sizes = new int[HIDDEN_LAYERS.length + 2];

        sizes[0] = N_FEATURES;

        for (int i = 0; i < HIDDEN_LAYERS.length; i++) {
            sizes[i + 1] = HIDDEN_LAYERS[i];
        }

        sizes[sizes.length - 1] = 1;

        System.out.println(
                "network layer sizes: " + Arrays.toString(sizes)
        );
        java.util.Random rng = new java.util.Random(SEED);
        Network model = initWeights(sizes, SEED);

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {

            // Randomize the order of mini-batches
            java.util.List<Integer> order = new java.util.ArrayList<>();

            for (int i = 0; i < nBatches; i++) {
                order.add(i);
            }

            java.util.Collections.shuffle(order, rng);

            for (int i : order) {

                if (batchSizes[i] == 0) {
                    continue;
                }

                // Broadcast the current model parameters
                org.apache.spark.broadcast.Broadcast<Network> bcW =
                        jsc.broadcast(model);

                // Map: calculate gradients for each sample
                // Reduce: sum gradients by layer index
                java.util.Map<Integer, Gradient> grad =
                        batches[i]
                                .flatMapToPair(
                                        r -> mapFn(r, bcW)
                                                .iterator()
                                )
                                .reduceByKey(Main::reduceFn)
                                .collectAsMap();

                // Update weights and biases on the driver
                for (int l = 0; l < model.weights.length; l++) {

                    Gradient g = grad.get(l);

                    if (g == null) {
                        continue;
                    }

                    double[][] dW = g.dW;
                    double[] db = g.db;

                    for (int j = 0;
                         j < model.weights[l].length; j++) {

                        for (int k = 0;
                             k < model.weights[l][j].length; k++) {

                            model.weights[l][j][k] -=
                                    ALPHA * dW[j][k] / batchSizes[i];
                        }

                        model.biases[l][j] -=
                                ALPHA * db[j] / batchSizes[i];
                    }
                }

                bcW.unpersist();
            }

            // ---- Training accuracy ----
            org.apache.spark.broadcast.Broadcast<Network> bcNow =
                    jsc.broadcast(model);

            java.util.Map<String, Long> counts =
                    trainRdd
                            .mapToPair(r -> predictCounts(r, bcNow))
                            .reduceByKey(Long::sum)
                            .collectAsMap();

            long tp = counts.getOrDefault("TP", 0L);
            long tn = counts.getOrDefault("TN", 0L);

            double accuracy =
                    nTrain > 0 ? (tp + tn) / (double) nTrain : 0.0;

            System.out.printf(
                    "epoch %3d  train accuracy: %.4f%n",
                    epoch, accuracy
            );

            bcNow.unpersist();
        }

        // ---- Test evaluation ----
        org.apache.spark.broadcast.Broadcast<Network> bcW =
                jsc.broadcast(model);

        java.util.Map<String, Long> counts =
                testRdd
                        .mapToPair(r -> predictCounts(r, bcW))
                        .reduceByKey(Long::sum)
                        .collectAsMap();

        long TP = counts.getOrDefault("TP", 0L);
        long FP = counts.getOrDefault("FP", 0L);
        long TN = counts.getOrDefault("TN", 0L);
        long FN = counts.getOrDefault("FN", 0L);

        long total = TP + FP + TN + FN;

        double accuracy =
                total > 0 ? (TP + TN) / (double) total : 0.0;

        double precision =
                TP + FP > 0 ? TP / (double) (TP + FP) : 0.0;

        double recall =
                TP + FN > 0 ? TP / (double) (TP + FN) : 0.0;

        double f1 =
                precision + recall > 0
                        ? 2 * precision * recall
                        / (precision + recall)
                        : 0.0;

        double specificity =
                TN + FP > 0 ? TN / (double) (TN + FP) : 0.0;

        System.out.println("\n=========== TEST RESULTS ===========");
        System.out.println("Confusion matrix (rows = actual, cols = predicted)");
        System.out.printf("%20s %10s %10s%n", "", "pred 0", "pred 1");
        System.out.printf("%20s %10d %10d%n", "actual 0", TN, FP);
        System.out.printf("%20s %10d %10d%n", "actual 1", FN, TP);

        System.out.printf("Accuracy   : %.4f%n", accuracy);
        System.out.printf("Precision  : %.4f%n", precision);
        System.out.printf("Recall     : %.4f%n", recall);
        System.out.printf("Specificity: %.4f%n", specificity);
        System.out.printf("F1 score   : %.4f%n", f1);

        bcW.unpersist(true);

        // ---- Stop Spark ----
        trainRdd.unpersist(true);
        testRdd.unpersist(true);

        for (JavaRDD<TrainingSamples> batch : batches) {
            batch.unpersist(true);
        }

        spark.stop();
    }
    // ----------------------------- TRAINING SAMPLES ---------------------------

    static class TrainingSamples implements Serializable {

        private static final long serialVersionUID = 1L;

        double[] x;
        double t;

        TrainingSamples(double[] x, double t) {
            this.x = x;
            this.t = t;
        }
    }

    static TrainingSamples toTrainingSamples(Row row, double[] mean, double[] std) {

        // Column 0 contains the label.
        double t = row.getDouble(0);

        // Columns 1 through 28 contain the features.
        double[] x = new double[N_FEATURES];

        for (int i = 0; i < N_FEATURES; i++) {
            double value = row.getDouble(i + 1);

            // Standardize each feature using training statistics.
            x[i] = (value - mean[i]) / std[i];
        }

        return new TrainingSamples(x, t);
    }

    // ----------------------------- NETWORK LAYER ---------------------------

    static class Network implements Serializable {

        private static final long serialVersionUID = 1L;

        double[][][] weights;
        double[][] biases;

        Network(double[][][] weights, double[][] biases) {
            this.weights = weights;
            this.biases = biases;
        }
    }
    static Network initWeights(int[] sizes, long seed) {

        Random rng = new Random(seed);

        int layers = sizes.length - 1;

        double[][][] W = new double[layers][][];
        double[][] b = new double[layers][];

        for (int l = 0; l < layers; l++) {

            int inputSize = sizes[l];
            int outputSize = sizes[l + 1];

            W[l] = new double[outputSize][inputSize];
            b[l] = new double[outputSize];

            double scale = Math.sqrt(inputSize);

            for (int j = 0; j < outputSize; j++) {
                for (int k = 0; k < inputSize; k++) {
                    W[l][j][k] = rng.nextGaussian() / scale;
                }
            }
        }

        return new Network(W, b);
    }

    // ---------------------------- SIGMOID ----------------------------
    static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    // -------------------------- FORWARD PASS -------------------------

    static double[][] forwardPass(double[] x, Network model) {

        int layers = model.weights.length;

        // acts[0] = input
        // acts[1] = first hidden layer
        // acts[layers] = output
        double[][] acts = new double[layers + 1][];

        acts[0] = x;

        for (int l = 0; l < layers; l++) {

            double[][][] W = model.weights;
            double[] b = model.biases[l];

            int outputSize = W[l].length;

            acts[l + 1] = new double[outputSize];

            for (int j = 0; j < outputSize; j++) {

                double z = b[j];

                for (int k = 0; k < acts[l].length; k++) {
                    z += W[l][j][k] * acts[l][k];
                }

                acts[l + 1][j] = sigmoid(z);
            }
        }

        return acts;
    }

    // ---------------------------- GRADIENT ---------------------------

    static class Gradient implements Serializable {

        private static final long serialVersionUID = 1L;

        double[][] dW;
        double[] db;

        Gradient(double[][] dW, double[] db) {
            this.dW = dW;
            this.db = db;
        }
    }

    // ------------------------------- MAP -----------------------------

    static List<Tuple2<Integer, Gradient>> mapFn(
            TrainingSamples record,
            Broadcast<Network> bcW) {

        double[] x = record.x;
        double t = record.t;

        Network model = bcW.value();

        double[][] a = forwardPass(x, model);

        double[] out = a[a.length - 1];

        double[] delta = new double[out.length];

        // Output-layer delta
        for (int k = 0; k < out.length; k++) {
            delta[k] = (out[k] - t)
                    * out[k]
                    * (1.0 - out[k]);
        }

        List<Tuple2<Integer, Gradient>> emitted =
                new ArrayList<>();

        double[][][] W = model.weights;

        // Traverse layers backward
        for (int l = W.length - 1; l >= 0; l--) {

            // dW = outer product(delta, previous activations)
            double[][] dW = new double[delta.length][a[l].length];

            for (int j = 0; j < delta.length; j++) {
                for (int k = 0; k < a[l].length; k++) {
                    dW[j][k] = delta[j] * a[l][k];
                }
            }

            double[] db = delta.clone();

            // Emit: (layer index, gradient)
            emitted.add(
                    new Tuple2<>(l, new Gradient(dW, db))
            );

            // Calculate the previous hidden layer's delta
            if (l > 0) {

                double[] previousDelta = new double[a[l].length];

                for (int j = 0; j < a[l].length; j++) {

                    double sum = 0.0;

                    for (int k = 0; k < delta.length; k++) {
                        sum += W[l][k][j] * delta[k];
                    }

                    previousDelta[j] =
                            a[l][j]
                                    * (1.0 - a[l][j])
                                    * sum;
                }

                delta = previousDelta;
            }
        }

        return emitted;
    }


    // ------------------------------ REDUCE ---------------------------

    static Gradient reduceFn(Gradient g1, Gradient g2) {

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

    // ----------------------------- EVALUATION ------------------------

    static Tuple2<String, Long> predictCounts(
            TrainingSamples record,
            Broadcast<Network> bcW) {

        double[] output =
                forwardPass(record.x, bcW.value())[
                        bcW.value().weights.length
                        ];

        int pred = output[0] >= 0.5 ? 1 : 0;
        int actual = record.t == 1.0 ? 1 : 0;

        String key;

        if (pred == 1 && actual == 1) {
            key = "TP";
        } else if (pred == 1) {
            key = "FP";
        } else if (actual == 1) {
            key = "FN";
        } else {
            key = "TN";
        }

        return new Tuple2<>(key, 1L);
    }




}
