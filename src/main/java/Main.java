import org.nd4j.linalg.api.ndarray.INDArray;

public class Main {

    public static void main(String[] args) throws Exception {
        BackgroundRemover b = BackgroundRemover.loadModel("/Users/tonykwok/shadow/experiment/image-background-removal/mobile_net_model/frozen_inference_graph.pb");
        INDArray p = b.predict("/Users/tonykwok/Downloads/hhhh.jpeg");
    }

}