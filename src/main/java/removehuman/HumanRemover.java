package removehuman;

import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;

public class HumanRemover {

    private static final int RESULT_INDEX = 4;
    private static final int PADDING = 10;
    private static final String DEFAULT_MODEL_FILE_PATH = "/etc/model/model.pb";
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

        for (int i = 0; i < shape[1]; ++i) {
            for (int j = 0; j < shape[2]; ++j) {
                int l = i >= PADDING ? result.getInt(0, i-PADDING, j) : 0;
                int r = i < shape[1]- PADDING ? result.getInt(0, i+PADDING, j) : 0;

                int u = j >= PADDING ? result.getInt(0, i, j-PADDING) : 0;
                int d = j < shape[2]- PADDING ? result.getInt(0, i, j+PADDING) : 0;

                int x = result.getInt(0, i, j);
                paddedResult.putScalar(new long[]{0, i, j}, x + l + r + u + d);

            }
        }
        return paddedResult;
    }

}