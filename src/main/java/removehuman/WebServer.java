package removehuman;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import removehuman.seamcarving.SeamCarvingUtils;
import spark.Request;
import spark.Response;
import spark.Spark;

import javax.servlet.MultipartConfigElement;
import javax.servlet.ServletException;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class WebServer {

    private static final Logger logger = LoggerFactory.getLogger(WebServer.class);
    private static final double INPUT_SIZE = 512.0d;
    private final HumanRemover model;

    public static void main(String[] args) {
        new WebServer().start();
    }

    public WebServer() {
        model = HumanRemover.loadModel(System.getenv("MODEL_PATH"));
    }

    private void start() {
        logger.info("Starting Web server");
        Spark.port(5000);
        Spark.staticFiles.location("/public"); // Static files
        Spark.post("/removehuman", this::inference);
        Spark.awaitInitialization();
        logger.info("Web server started at {}", Spark.port());
    }

    private Object inference(Request request, Response response) throws IOException {
        byte[] body = getImageBytes(request);
        logger.info("Received bytes with length {}", body.length);
        long start = System.currentTimeMillis();
        Mat resizedImageMat = convertToMat(body);
        INDArray input = cvMatToINDArrayResized(resizedImageMat);
        INDArray predicted = predictHumanMask(input);
        Mat bufferedImage = SeamCarvingUtils.removeHumanFromImage(resizedImageMat, predicted);
        BytePointer outputPointer = new BytePointer();
        imencode(".jpg", bufferedImage, outputPointer);
        byte[] outputBuffer = new byte[(int) outputPointer.limit()];
        outputPointer.get(outputBuffer);
        response.raw().setContentType("image/jpeg");
        logger.info("Took {}ms to finish all steps", System.currentTimeMillis() - start);
        return outputBuffer;
    }

    private byte[] getImageBytes(Request request) throws IOException {
        request.attribute("org.eclipse.jetty.multipartConfig", new MultipartConfigElement("/temp"));
        try (InputStream is = request.raw().getPart("file").getInputStream()) {
            byte[] body = new byte[is.available()];
            is.read(body);
            logger.info("Get image data from form data");
            return body;
        } catch (ServletException e) {
            byte[] bytes = request.bodyAsBytes();
            logger.info("Get image data from body bytes");
            return bytes;
        }
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
        INDArray result = model.predict(input);
        logger.info("Took {}ms to finish predicting segment from model", System.currentTimeMillis() - start);
        return result;
    }

    private static INDArray cvMatToINDArrayResized(Mat mat) throws IOException {
        long start = System.currentTimeMillis();
        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        INDArray indArray = nativeImageLoader.asMatrix(mat).permute(0, 2, 3, 1);
        logger.info("Took {}ms to finish converting image to nd array", System.currentTimeMillis() - start);
        return indArray;
    }


}