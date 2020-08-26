package removebg;

import org.bytedeco.javacv.Java2DFrameUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import spark.Request;
import spark.Response;
import spark.Spark;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.OutputStream;

import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_UNCHANGED;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imdecode;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_photo.INPAINT_TELEA;
import static org.bytedeco.opencv.global.opencv_photo.inpaint;

public class RemoveBackgroundWebServer {

    private static final double INPUT_SIZE = 512.0d;
    private final BackgroundRemover b = BackgroundRemover.loadModel(System.getenv("MODEL_PATH"));

    public static void main(String[] args) {
        new RemoveBackgroundWebServer().start();
    }

    private void start() {
        Spark.port(5000);
        Spark.post("/removebg", this::inference);
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
        INDArray input = matToIndArray(resizedImageMat);
        INDArray predicted = predict(input);
        BufferedImage bufferedImage = drawSegment(resizedImageMat, predicted);
        response.raw().setContentType("image/png");
        try (OutputStream out = response.raw().getOutputStream()) {
            ImageIO.write(bufferedImage, "png", out);
        }

        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish all steps");
        return response;
    }

    private Mat convertToMat(byte[] body) {
        Mat baseImgMat = imdecode(new Mat(body), IMREAD_UNCHANGED);
        double resizeRatio = INPUT_SIZE / Math.max(baseImgMat.cols(), baseImgMat.rows());
        while (resizeRatio > 1.0d) {
            resizeRatio /= 2;
        }
        Size size = new Size((int) (baseImgMat.cols() * resizeRatio), (int) (baseImgMat.rows() * resizeRatio));
        Mat mat = new Mat(size);
        resize(baseImgMat, mat, size);
        return mat;
    }

    private INDArray predict(INDArray input) {
        long start = System.currentTimeMillis();
        INDArray result = b.predict(input);
        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish predicting segment from model");
        return result;
    }

    private static INDArray matToIndArray(Mat mat) throws IOException {
        long start = System.currentTimeMillis();
        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        INDArray indArray = nativeImageLoader.asMatrix(mat).permute(0, 2, 3, 1);
        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish converting image to nd array");
        return indArray;
    }


    private static BufferedImage drawSegment(Mat baseImg, INDArray matImg) {
        long start = System.currentTimeMillis();

        int height = baseImg.rows();
        int width = baseImg.cols();

        BufferedImage maskImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int mask = matImg.getInt(0, y, x);
                if (mask != 0) {
                    maskImage.setRGB(x, y, new Color(255, 255, 255).getRGB());
                } else {
                    maskImage.setRGB(x, y, new Color(0, 0, 0).getRGB());
                }

            }
        }

        Mat maskMat = Java2DFrameUtils.toMat(maskImage);
        inpaint(baseImg, maskMat, baseImg, 1.0d, INPAINT_TELEA);
        BufferedImage resultImage = Java2DFrameUtils.toBufferedImage(baseImg);
        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish drawing output image");
        return resultImage;
    }

}