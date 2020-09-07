package removehuman;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import removehuman.seamcarving.SeamCarvingUtils;
import spark.Request;
import spark.Response;
import spark.Spark;

import java.io.IOException;
import java.nio.ByteBuffer;

import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class WebServer {

    private static final double INPUT_SIZE = 512.0d;

    private final HumanRemover b = HumanRemover.loadModel(System.getenv("MODEL_PATH"));

    public static void main(String[] args) {
        new WebServer().start();
    }

    private void start() {
        Spark.port(5000);
        Spark.post("/removehuman", this::inference);
        Spark.awaitInitialization();
        System.out.println("Started");
    }

    private Object inference(Request request, Response response) throws IOException {
        try {
            return doInference(request, response);
        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        }
    }

    private Object doInference(Request request, Response response) throws IOException {
        byte[] body = request.bodyAsBytes();
        System.out.println("Received bytes with length " + body.length);
        long start = System.currentTimeMillis();
        Mat resizedImageMat = convertToMat(body);
        INDArray input = cvMatToINDArrayResized(resizedImageMat);
        INDArray predicted = predictHumanMask(input);
        Mat bufferedImage = SeamCarvingUtils.removeHumanFromImage(resizedImageMat, predicted);
        ByteBuffer buffer = ByteBuffer.allocate(1000000); //TODO: use pool / thread local
        imencode(".jpg", bufferedImage, buffer);
        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish all steps");
        response.raw().setContentType("image/jpeg");
        return buffer;
    }


    private Mat convertToMat(byte[] body) {
        Mat baseImgMat = imdecode(new Mat(body), IMREAD_COLOR);
        double resizeRatio = INPUT_SIZE / Math.max(baseImgMat.cols(), baseImgMat.rows());
        while (resizeRatio > 1.0d) {
            resizeRatio /= 2;
        }
        Size size = new Size((int) (baseImgMat.cols() * resizeRatio), (int) (baseImgMat.rows() * resizeRatio));
        Mat mat = new Mat(size);
        resize(baseImgMat, mat, size);
        return mat;
    }

    private INDArray predictHumanMask(INDArray input) {
        long start = System.currentTimeMillis();
        INDArray result = b.predict(input);
        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish predicting segment from model");
        return result;
    }

    private static INDArray cvMatToINDArrayResized(Mat mat) throws IOException {
        long start = System.currentTimeMillis();
        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        INDArray indArray = nativeImageLoader.asMatrix(mat).permute(0, 2, 3, 1);
        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish converting image to nd array");
        return indArray;
    }




}