import java.io.Serializable;

public class Gradient implements Serializable {
    private static final long serialVersionUID = 1L;

    double[][] dW;
    double[] db;

    public Gradient(double[][] dW, double[] db) {
        this.dW = dW;
        this.db = db;
    }
}

