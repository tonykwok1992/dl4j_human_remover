package removehuman.seamcarving;

import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacv.Java2DFrameUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Range;
import org.bytedeco.opencv.opencv_core.Rect;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import removehuman.AnimationGenerator;

import javax.imageio.ImageIO;

import java.io.File;
import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class SeamCarvingUtils {

    private static final int NO_IMPROVEMENT_COUNT_BREAK = 10;
    private static final INDArrayIndex ALL_INDEX = NDArrayIndex.all();

    public static Mat removeHumanFromImage(Mat baseImg, INDArray maskArea) {
        long start = System.currentTimeMillis();
        AnimationGenerator animationGenerator = new AnimationGenerator();
        int originalCols = baseImg.cols();
        int maxWidthToRemove = originalCols / 4; //Assume you cannot cut more than one-forth of width from photos

        Mat energy = computeEnergyMatrixWithMask(baseImg, maskArea);
        Recorder record = new Recorder(NO_IMPROVEMENT_COUNT_BREAK);
        for (int i = 0; i < maxWidthToRemove ; i++) {
            if(!record.record(Nd4j.getExecutioner().exec(new CountNonZero(maskArea)).getInt(0))){
                break;
            }
            INDArray seam = findVerticalSeam(baseImg, energy);
            removeVerticalSeam(baseImg, maskArea, seam);
            baseImg = decrementWidthByOne(baseImg);
            maskArea = decrementWidthByOne(maskArea);
            energy = computeEnergyMatrixWithMask(baseImg, maskArea);

            animationGenerator.recordFrame(baseImg);

        }

        Mat imgOut = baseImg.clone();

        int toAddCount = originalCols - baseImg.cols();
        for (int i = 0; i < toAddCount; i++) {
            INDArray seam = findVerticalSeam(baseImg, energy);
            removeVerticalSeam(baseImg, maskArea, seam);
            baseImg = decrementWidthByOne(baseImg);
            imgOut = addVerticalSeam(imgOut, seam, i);
            energy = computeEnergyMatrix(baseImg);

            animationGenerator.recordFrame(imgOut);
        }

        System.out.println("Took " + (System.currentTimeMillis() - start) + "ms to finish removing human from image");
        return imgOut;
    }

    private static INDArray decrementWidthByOne(INDArray maskArea) {
        return maskArea.get(ALL_INDEX, ALL_INDEX, NDArrayIndex.interval(0, maskArea.shape()[2] - 1));
    }

    private static Mat decrementWidthByOne(Mat baseImg) {
        return baseImg.apply(new Range(0, baseImg.rows()), new Range(0, baseImg.cols() - 1));
    }

    private static Mat addVerticalSeam(Mat imgOut, INDArray seam, int numIter) {
        int rows = imgOut.rows();
        int cols = imgOut.cols();
        int channels = imgOut.channels();

        Mat imgExtend = new Mat(imgOut.rows(), imgOut.cols() + 1, imgOut.type());
        Mat roiImgExtend = new Mat(imgExtend, new Rect(0, 0, imgOut.cols(), imgOut.rows()));
        imgOut.copyTo(roiImgExtend);

        UByteIndexer indexer = imgExtend.createIndexer();
        seam = seam.add(numIter);

        for (int row = 0; row < rows; row++) {
            int seamInt = seam.getInt(row);
            if(seamInt == 0) continue;
            if(seamInt == cols-1) continue;
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
