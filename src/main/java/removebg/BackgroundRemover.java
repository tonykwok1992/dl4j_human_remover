package removebg;

import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.io.File;

public class BackgroundRemover {

    private final SameDiff sd;

    public static BackgroundRemover loadModel(String file) {
        return new BackgroundRemover(TFGraphMapper.importGraph(new File(file)));
    }

    private BackgroundRemover(SameDiff sd){
        this.sd = sd;
    }

    public INDArray predict (INDArray indArray) {
        sd.associateArrayWithVariable(indArray, sd.variables().get(0));
        NativeGraphExecutioner executioner = new NativeGraphExecutioner();
        INDArray[] results = executioner.executeGraph(sd);
        return results[4];
    }
}