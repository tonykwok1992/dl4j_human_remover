package removebg;

import org.bytedeco.javacv.Java2DFrameUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import spark.Request;
import spark.Response;
import spark.Spark;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

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
        try (InputStream bio = new ByteArrayInputStream(body)) {
            BufferedImage bimg = ImageIO.read(bio);
            INDArray input = readStreamToBufferedImage(bimg);
            INDArray mat = predict(input);
            BufferedImage bufferedImage = drawSegment(bimg, mat);
            response.raw().setContentType("image/png");
            try (OutputStream out = response.raw().getOutputStream()) {
                ImageIO.write(bufferedImage, "png", out);
            }
        }
        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish all steps");
        return response;
    }

    private INDArray predict(INDArray input) {
        long start = System.currentTimeMillis();
        INDArray result = b.predict(input);
        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish predicting segment from model");
        return result;
    }

    private INDArray readStreamToBufferedImage(BufferedImage bimg) throws IOException {
        long start = System.currentTimeMillis();
        int width = bimg.getWidth();
        int height = bimg.getHeight();
        double resizeRatio = INPUT_SIZE / Math.max(width, height);
        while (resizeRatio > 1.0d) {
            resizeRatio /= 2;
        }
        Java2DNativeImageLoader l = new Java2DNativeImageLoader((int) (height * resizeRatio), (int) (width * resizeRatio), 3);
        INDArray indArray = l.asMatrix(bimg).permute(0, 2, 3, 1);
        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish converting image to nd array");
        return indArray;
    }


    private static BufferedImage drawSegment(BufferedImage baseImg, INDArray matImg) {
        long start = System.currentTimeMillis();

        int height = baseImg.getHeight();
        int width = baseImg.getWidth();

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
        Mat imageMat = Java2DFrameUtils.toMat(baseImg);
        inpaint(imageMat, maskMat, imageMat, 1.0d, INPAINT_TELEA);
        BufferedImage resultImage = Java2DFrameUtils.toBufferedImage(imageMat);
        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish drawing output image");
        return resultImage;
    }

}