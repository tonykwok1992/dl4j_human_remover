package removehuman;

import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;

public class HumanRemover {

    private static final int RESULT_INDEX = 4;
    private static final int PADDING = 10;
    private static final String DEFAULT_MODEL_FILE_PATH = "/usr/local/model/model.pb";
    private final SameDiff sd;

    public static HumanRemover loadModel(String file) {
        if (file == null) {
            file = DEFAULT_MODEL_FILE_PATH;
        }
        return new HumanRemover(TFGraphMapper.importGraph(new File(file)));
    }

    private HumanRemover(SameDiff sd) {
        this.sd = sd;
    }

    public INDArray predict(INDArray indArray) {
        sd.associateArrayWithVariable(indArray, sd.variables().get(0));
        NativeGraphExecutioner executioner = new NativeGraphExecutioner();
        INDArray[] results = executioner.executeGraph(sd);
        INDArray result = results[RESULT_INDEX];
        INDArray paddedResult = padResult(result);
        return paddedResult;
    }

    //The predicated segment often leave some edges not contained, padding the result to reduce error.
    private INDArray padResult(INDArray result) {
        long[] shape = result.shape();
        INDArray paddedResult = result.like();

        long rows = shape[1];
        long cols = shape[2];
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                int l = row >= PADDING ? result.getInt(0, row - PADDING, col) : 0;
                int r = row < rows - PADDING ? result.getInt(0, row + PADDING, col) : 0;

                int u = col >= PADDING ? result.getInt(0, row, col - PADDING) : 0;
                int d = col < cols - PADDING ? result.getInt(0, row, col + PADDING) : 0;

                int x = result.getInt(0, row, col);
                paddedResult.putScalar(new long[]{0, row, col}, x + l + r + u + d);

            }
        }
        return paddedResult;
    }

}