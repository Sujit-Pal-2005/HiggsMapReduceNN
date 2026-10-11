import java.io.Serializable;

public class Network implements Serializable {
    private static final long serialVersionUID = 1L;

    double[][][] weights;
    double[][] biases;

    public Network(double[][][] weights, double[][] biases) {
        this.weights = weights;
        this.biases = biases;
    }
}