import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;

import java.io.File;

import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

public class BackgroundRemover {

    private static final double INPUT_SIZE = 512.0d;
    private final SameDiff sd;

    public static BackgroundRemover loadModel(String file) {
        return new BackgroundRemover(TFGraphMapper.importGraph(new File(file)));
    }

    private BackgroundRemover(SameDiff sd){
        this.sd = sd;
    }

    public INDArray predict (String filepath) throws IOException{
        File file = new File(filepath);
        BufferedImage bimg = ImageIO.read(file);
        int width          = bimg.getWidth();
        int height         = bimg.getHeight();
        double resizeRatio = INPUT_SIZE / Math.max(width, height);
        NativeImageLoader l = new NativeImageLoader((long) (height*resizeRatio), (long) (width*resizeRatio),3);
        INDArray indArray = l.asMatrix(file, false);
        sd.associateArrayWithVariable(indArray, sd.variables().get(0));
        NativeGraphExecutioner executioner = new NativeGraphExecutioner();
        INDArray[] results = executioner.executeGraph(sd); //returns an array of the outputs
        BufferedImage bufferedImage = drawSegment(indArray, results[4]);
        File outputfile = new File("image.jpg");
        ImageIO.write(bufferedImage, "jpg", outputfile);
        return null;
    }

    private BufferedImage drawSegment(INDArray baseImg,INDArray matImg) {
        long[] shape = baseImg.shape();

        long height = shape[1];
        long width = shape[2];
        BufferedImage image = new BufferedImage((int)width, (int)height, BufferedImage.TYPE_INT_RGB);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int mask = matImg.getInt(0, y, x);
                if (mask != 0) {
                    int red = baseImg.getInt(0,  y, x,2);
                    int green = baseImg.getInt(0, y, x,1);
                    int blue = baseImg.getInt(0, y, x,0);

                    red = Math.max(Math.min(red, 255), 0);
                    green = Math.max(Math.min(green, 255), 0);
                    blue = Math.max(Math.min(blue, 255), 0);
                    image.setRGB(x, y, new Color(red, green, blue).getRGB());
                }

            }
        }
        return image;
    }
}