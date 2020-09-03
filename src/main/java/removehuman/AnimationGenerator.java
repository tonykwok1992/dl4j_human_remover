package removehuman;

import org.bytedeco.javacv.Java2DFrameUtils;
import org.bytedeco.opencv.opencv_core.Mat;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;

public class AnimationGenerator {

    private int counter = 0;

    public void recordFrame(Mat img) {
        try {
            ImageIO.write(Java2DFrameUtils.toBufferedImage(img), "jpg", new File("/tmp/image" + String.format("%04d" , counter++) + ".jpg"));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

}
