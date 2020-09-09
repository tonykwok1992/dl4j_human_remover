package removehuman;

import org.bytedeco.javacv.Java2DFrameUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;

public class AnimationGenerator {

    private static final Logger logger = LoggerFactory.getLogger(AnimationGenerator.class);
    private static final String ANIMATION_ENABLED_ENV = System.getenv("ANIMATION_ENABLED");
    private static final boolean ANIMATION_ENABLED = "Y".equals(ANIMATION_ENABLED_ENV);
    private int counter = 0;

    public void recordFrame(Mat img) {
        if(!ANIMATION_ENABLED){
            return;
        }

        try {
            ImageIO.write(Java2DFrameUtils.toBufferedImage(img), "jpg", new File("/tmp/image" + String.format("%04d" , counter++) + ".jpg"));
        } catch (IOException e) {
            logger.error("Error when writing frame for animation.", e);
        }
    }

}
