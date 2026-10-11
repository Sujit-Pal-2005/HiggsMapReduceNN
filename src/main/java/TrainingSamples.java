import org.apache.spark.sql.Row;
import java.io.Serializable;

public class TrainingSamples implements Serializable {
    private static final long serialVersionUID = 1L;

    final double[] x;
    final double t;

    public TrainingSamples(double[] x, double t) {
        this.x = x;
        this.t = t;
    }

    public static TrainingSamples fromRow(Row row, double[] mean, double[] std, int nFeatures) {
        double label = row.getDouble(0);
        double[] features = new double[nFeatures];

        for (int i = 0; i < nFeatures; i++) {
            double value = row.getDouble(i + 1);
            features[i] = (value - mean[i]) / std[i];
        }

        return new TrainingSamples(features, label);
    }
}
