package removebg.inpaint;

import java.awt.image.BufferedImage;

public class FastRGB {

    private final BufferedImage image;
    public int width;
    public int height;

    FastRGB(BufferedImage image) {
        this.image = image;
        width = image.getWidth();
        height = image.getHeight();
    }

    void setRGB(int x, int y, int rgb) {
        image.setRGB(x, y, rgb);
    }

    int getRGB(int x, int y) {
        return image.getRGB(x, y);
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public void decrementHeight() {
        height--;
    }

    public void decrementWidth() {
        width--;
    }

    public BufferedImage getImg() {
        return image;
    }

}