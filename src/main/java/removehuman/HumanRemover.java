package removehuman;

import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;

public class HumanRemover {

    private static final int RESULT_INDEX = 4;
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
        return result;
    }

}