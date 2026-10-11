import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.stddev;
import static org.apache.spark.sql.functions.col;

public class Main2{

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
                .map(row -> TrainingSamples.fromRow(row, mean, std, N_FEATURES))
                .cache();

        JavaRDD<TrainingSamples> testRdd = testDf.javaRDD()
                .map(row -> TrainingSamples.fromRow(row, mean, std, N_FEATURES))
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
        Network model = NeuralNetwork.initWeights(sizes, SEED);

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
                                        r -> NeuralNetwork.mapFn(r, bcW)
                                                .iterator()
                                )
                                .reduceByKey(NeuralNetwork::reduceFn)
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
                            .mapToPair(r -> NeuralNetwork.predictCounts(r, bcNow))
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
                        .mapToPair(r -> NeuralNetwork.predictCounts(r, bcW))
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
}