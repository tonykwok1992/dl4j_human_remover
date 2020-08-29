package removebg;

import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Range;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
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
    private static final int MAX_WIDTH_TO_REMOVE = (int) (INPUT_SIZE / 2);
    private static final int NO_IMPROVEMENT_COUNT_BREAK = 5;
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
        INDArray input = cvMatToINDArrayResized(resizedImageMat);
        INDArray predicted = predictHumanMask(input);
        Mat bufferedImage = removeHumanFromImage(resizedImageMat, predicted);
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


    private static Mat removeHumanFromImage(Mat baseImg, INDArray maskArea) {
        long start = System.currentTimeMillis();
        final INDArrayIndex allIndex = NDArrayIndex.all();
        int oriRow = baseImg.cols();
        int lastNonZeroCount = Integer.MAX_VALUE;
        int noImproveCount = 0;
        Mat energy = null;
        for (int i = 0; i < MAX_WIDTH_TO_REMOVE && noImproveCount <= NO_IMPROVEMENT_COUNT_BREAK; i++) {
            int currentNonZeroCount = Nd4j.getExecutioner().exec(new CountNonZero(maskArea)).getInt(0);
            System.out.println("lastNonZeroCount: " + lastNonZeroCount);
            System.out.println("noImproveCount: " + noImproveCount);
            if (lastNonZeroCount == currentNonZeroCount) {
                noImproveCount++;
            } else {
                noImproveCount = 0;
            }
            energy = computeEnergyMatrixWithMask(baseImg, maskArea);
            INDArray seam = findVerticalSeam(baseImg, energy);
            removeVerticalSeam(baseImg, maskArea, seam);
            baseImg = baseImg.apply(new Range(0, baseImg.rows()), new Range(0, baseImg.cols() - 1));
            maskArea = maskArea.get(allIndex, allIndex, NDArrayIndex.interval(0, baseImg.cols() - 1));

            lastNonZeroCount = currentNonZeroCount;
        }

        Mat imgOut = baseImg.clone();

        int toAddCount = oriRow - baseImg.cols();
        for (int i = 0; i < toAddCount; i++) {
            INDArray seam = findVerticalSeam(baseImg, energy);
            removeVerticalSeam(baseImg, maskArea, seam);
            baseImg = baseImg.apply(new Range(0, baseImg.rows()), new Range(0, baseImg.cols() - 1));

            imgOut = addVerticalSeam(imgOut, seam, i);
            energy = computeEnergyMatrix(baseImg);

        }


        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish removing human from image");
        return imgOut;
    }

    private static Mat addVerticalSeam(Mat baseImg, INDArray seam, int numIter) {
        int rows = baseImg.rows();
        int cols = baseImg.cols();
        int channels = baseImg.channels();

        Mat imgExtend = new Mat(baseImg.rows(), baseImg.cols()+1, baseImg.type());
        UByteIndexer baseImgIndexer = baseImg.createIndexer();
        UByteIndexer indexer = imgExtend.createIndexer();

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                for (int channel = 0; channel < channels; channel++) {
                    indexer.put(new long[]{row, col, channel}, baseImgIndexer.get(row, col, channel) );
                }
            }
        }

        seam = seam.add(numIter);

        for (int row = 0; row < rows; row++) {
            int seamInt = seam.getInt(row);
            for (int col = cols; col > seamInt; col--) {
                for (int channel = 0; channel < channels; channel++) {
                    indexer.put(new long[]{row, col, channel}, indexer.get(row, col - 1, channel) );
                }
            }

            for (int channel = 0; channel < channels; channel++) {
                int v1 = indexer.get(row, seamInt - 1, channel);
                int v2 = indexer.get(row, seamInt + 1, channel);
                indexer.put(new long[]{row, seamInt, channel}, (v1 + v2) / 2);
            }

        }
        return imgExtend;
    }

    private static void removeVerticalSeam(Mat baseImg, INDArray maskArea, INDArray seam) {
        int rows = baseImg.rows();
        int cols = baseImg.cols();
        int channels = baseImg.channels();

        UByteIndexer indexer = baseImg.createIndexer();
        for (int row = 0; row < rows; row++) {
            for (int col = seam.getInt(row); col < cols - 1; col++) {
                for (int channel = 0; channel < channels; channel++) {
                    indexer.put(row, col, channel, indexer.get(row, col + 1, channel));
                    maskArea.putScalar(new long[]{0, row, col}, maskArea.getDouble(new long[]{0, row, col + 1}));
                }
            }
        }
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

    public static Mat computeEnergyMatrixWithMask(Mat img, INDArray objectArea) {
        Mat energyMatrix = computeEnergyMatrix(img);
        UByteIndexer indexer = energyMatrix.createIndexer();

        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                int mask = objectArea.getInt(0, i, j);
                if (mask != 0) {
                    indexer.put(i, j, 0);
                }
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