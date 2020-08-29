package removebg;

import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Range;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.bool.Any;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountZero;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import spark.Request;
import spark.Response;
import spark.Spark;

import java.io.IOException;
import java.nio.ByteBuffer;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

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

        Mat bufferedImage = drawSegment(resizedImageMat, predicted);
        ByteBuffer buffer = ByteBuffer.allocate(1000000); //TODO: use pool / thread local
        imencode(".png", bufferedImage, buffer);
        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish all steps");
        response.raw().setContentType("image/png");
        return buffer;
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


    private static Mat drawSegment(Mat baseImg, INDArray objectArea) {
        long start = System.currentTimeMillis();
        for (int i = 0; i < 512 && objectArea.any(); i++) {
            Mat energy = computeEnergyMatrixModified(baseImg, objectArea);
            INDArray seam = findVerticalSeam(baseImg, energy);
            baseImg = removeVerticalSeam(baseImg, seam);
            objectArea = removeVerticalSeamFromMask(objectArea, seam);
        }

        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish drawing output image");
        return baseImg;
    }

    private static Mat removeVerticalSeam(Mat baseImg, INDArray seam) {
        int rows = baseImg.rows();
        int cols = baseImg.cols();
        int channels = baseImg.channels();

        UByteIndexer indexer = baseImg.createIndexer();
        for (int row = 0; row < rows; row++) {
            for (int col = seam.getInt(row); col < cols - 1; col++) {
                for (int channel = 0; channel < channels; channel++) {
                    indexer.put(row, col, channel, indexer.get(row, col + 1, channel));
                }
            }
        }
        return baseImg.apply(new Range(0, baseImg.rows()), new Range(0, baseImg.cols() - 1));
    }

    private static INDArray removeVerticalSeamFromMask(INDArray baseImg, INDArray seam) {
        int rows = baseImg.get(NDArrayIndex.point(0)).rows();
        int cols = baseImg.get(NDArrayIndex.point(0)).columns();

        for (int row = 0; row < rows; row++) {
            for (int col = seam.getInt(row); col < cols - 1; col++) {
                baseImg.putScalar(new long[]{0, row, col}, baseImg.getDouble(new long[]{0, row, col + 1}));
            }
        }

        return baseImg.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, cols - 1));
    }

    public static Mat computeEnergyMatrix(Mat img) {
        Mat gray = new Mat();
        Mat sobelX = new Mat();
        Mat sobelY = new Mat();

        Mat absSobelX = new Mat();
        Mat absSobelY = new Mat();

        Mat energyMatrix = new Mat();

        cvtColor(img, gray, CV_BGR2GRAY);
        Sobel(gray, sobelX, CV_64F, 1, 0);
        Sobel(gray, sobelY, CV_64F, 0, 1);
        convertScaleAbs(sobelX, absSobelX);
        convertScaleAbs(sobelY, absSobelY);
        addWeighted(absSobelX, 0.5, absSobelY, 0.5, 0, energyMatrix);
        return energyMatrix;
    }

    public static Mat computeEnergyMatrixModified(Mat img, INDArray objectArea) {
        Mat energyMatrix = computeEnergyMatrix(img);
        UByteIndexer indexer = energyMatrix.createIndexer();

        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                int mask = objectArea.getInt(0, i, j);
                if (mask != 0) {
                    indexer.put(i, j, 0);
                }
//                else{
//                    indexer.put(i, j, 1);
//
//                }
            }
        }
        return energyMatrix;
    }

    public static INDArray findVerticalSeam(Mat img, Mat energy) {
        int rows = img.rows();
        int cols = img.cols();
        INDArray distTo = Nd4j.zeros(rows, cols).assign(Double.MAX_VALUE);
        distTo.putRow(0, Nd4j.zeros(cols));
        INDArray edgeTo = Nd4j.zeros(rows, cols);

        Indexer indexer = energy.createIndexer();

        for (int row = 0; row < rows - 1; row++) {
            for (int col = 0; col < cols; col++) {
                if (col != 0) {
                    if (distTo.getDouble(row + 1, col - 1) > distTo.getDouble(row, col) + indexer.getDouble(row + 1, col - 1)) {
                        distTo.put(row + 1, col - 1, distTo.getDouble(row, col) + indexer.getDouble(row + 1, col - 1));
                        edgeTo.put(row + 1, col - 1, 1.0);
                    }
                }

                if (distTo.getDouble(row + 1, col) > distTo.getDouble(row, col) + indexer.getDouble(row + 1, col)) {
                    distTo.put(row + 1, col, distTo.getDouble(row, col) + indexer.getDouble(row + 1, col));
                    edgeTo.put(row + 1, col, 0.0);
                }

                if (col != cols - 1) {
                    if (distTo.getDouble(row + 1, col + 1) > distTo.getDouble(row, col) + indexer.getDouble(row + 1, col + 1)) {
                        distTo.put(row + 1, col + 1, distTo.getDouble(row, col) + indexer.getDouble(row + 1, col + 1));
                        edgeTo.put(row + 1, col + 1, -1.0d);
                    }
                }


            }
        }

        INDArray seam = Nd4j.zeros(rows);
        seam.put(rows - 1, Nd4j.argMin(distTo.getRow(rows - 1), 0));

        for (int i = rows - 1; i > 0; i--) {
            seam.putScalar(i - 1, seam.getDouble(i) + edgeTo.getDouble(i, seam.getInt(i)));
        }
        return seam;
    }


}