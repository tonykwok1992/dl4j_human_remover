package removebg.inpaint;

/**
 * This class implements Seam. A seam is a fixed size list of pixels that has a direction,
 * meaning it can either be horizontal or vertical.
 */
public class Seam {

    private double energy;
    private int[] pixels;
    private String direction;
    private int size;

    /**
     * Default constructor of the class.
     * @param s Indicates the number of pixels the seam has.
     * @param dir Indicates the direction of the seam, can be either "horizontal" or "vertical".
     */
    public Seam(int s, String dir)
    {
        this.size = s;
        pixels = new int[s];
        direction = dir;
    }

    /**
     * Accessor method for the direction of the seam.
     * @return Direction of the seam, either "horizontal" or "vertical".
     */
    String getDirection()
    {
        return direction;
    }

    /**
     * Accessor method for the list of pixels of the seam.
     * @return Integer array indicating the column (horizontal seam) or row (if vertical seam) of the pixels.
     */
    int[] getPixels()
    {
        return pixels;
    }

    /**
     * Mutator method for the pixels in the seam.
     * @param position Array position that will be set.
     * @param value Value that will be stored.
     */
    void setPixels(int position, int value)
    {
        pixels[position] = value;
    }

    /**
     * Accessor method for the energy of the seam.
     * @return Energy of the seam.
     */
    double getEnergy()
    {
        return energy;
    }

    /**
     * Mutator method for the energy of the seam.
     * @param energy Energy that will be set.
     */
    void setEnergy(double energy)
    {
        this.energy = energy;
    }

    /**
     * Accessor method for the size of the seam.
     * @return Integer value indicating the size/length of the seam.
     */
    int getSize()
    {
        return size;
    }
}