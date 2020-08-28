package removebg.inpaint;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

/**
 * This class implements a SeamCarver object, that takes an input image and uses the seam carving
 * technique to change the dimensions of the image.
 */
public class SeamCarver {

    private final FastRGB carvedImage;

    public SeamCarver(BufferedImage inputImage) {
        carvedImage = new FastRGB(inputImage);
    }

    public BufferedImage carveImage(int width, int height) throws IllegalArgumentException {
        // get the number of horizontal and vertical seams that we need to remove
        int num_carve_horizontal = carvedImage.getHeight() - height;
        int num_carve_vertical = carvedImage.getWidth() - width;

        int total_carve = num_carve_horizontal + num_carve_vertical;

        // if the dimensions of the output image is larger than the input, throw an exception
        if (num_carve_horizontal < 0 || num_carve_horizontal >= carvedImage.getWidth() || num_carve_vertical < 0 || num_carve_vertical >= carvedImage.getHeight()) {
            System.out.println("Width and height of the output image should be less then or equal to the original!");
            throw new IllegalArgumentException();
        }


        // Remove seams until we reach the output dimensions
        while (num_carve_horizontal > 0 || num_carve_vertical > 0) {
            Seam horizontal_seam;
            Seam vertical_seam;

            // get the current energy table
            double[][] energy_table = calculateEnergyTable(carvedImage);

            // if we can remove either horizontal or vertical seam
            // compare their energy values, remove the one with the minimum
            if (num_carve_horizontal > 0 && num_carve_vertical > 0) {
                horizontal_seam = getHorizontalSeam(energy_table);
                vertical_seam = getVerticalSeam(energy_table);

                if (horizontal_seam.getEnergy() < vertical_seam.getEnergy()) {
                    removeSeam(horizontal_seam);
                    num_carve_horizontal--;
                } else {
                    removeSeam(vertical_seam);
                    num_carve_vertical--;
                }
            }

            // if we should remove horizontal seam, remove it
            else if (num_carve_horizontal > 0) {
                horizontal_seam = getHorizontalSeam(energy_table);
                removeSeam(horizontal_seam);
                num_carve_horizontal--;
            }

            // if we should remove vertical seam, remove it
            else {
                vertical_seam = getVerticalSeam(energy_table);
                removeSeam(vertical_seam);
                num_carve_vertical--;
            }

        }
        return carvedImage.getImg().getSubimage(0, 0, carvedImage.width, carvedImage.height);
    }

    /**
     * This method takes and image and calculates energy of the each pixel using dual energy
     * gradient.
     *
     * @param image The image whose energy table will be calculated.
     * @return Double type 2d array with same dimensions as the input image.
     */
    private double[][] calculateEnergyTable(FastRGB image) {
        int width = image.getWidth();
        int height = image.getHeight();
        double[][] energyTable = new double[width][height];

        // loop over each pixel
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                double xEnergy, yEnergy, totalEnergy;

                // get next and previous horizontal pixels
                int xPrevRGB = image.getRGB((i - 1 + width) % width, j);
                int xNextRGB = image.getRGB((i + 1 + width) % width, j);

                // calculate the horizontal energy
                xEnergy = getEnergy(xPrevRGB, xNextRGB);

                // get next and previous vertical pixels
                int yPrevRGB = image.getRGB(i, (j - 1 + height) % height);
                int yNextRGB = image.getRGB(i, (j + 1 + height) % height);

                // calculate the vertical energy
                yEnergy = getEnergy(yPrevRGB, yNextRGB);

                // get the total energy
                totalEnergy = xEnergy + yEnergy;
                energyTable[i][j] = totalEnergy;
            }
        }

        return energyTable;
    }

    private double getEnergy(int rgb1, int rgb2) {
        // get r, g, b values of rgb1
        double b1 = (rgb1) & 0xff;
        double g1 = (rgb1 >> 8) & 0xff;
        double r1 = (rgb1 >> 16) & 0xff;

        // get r, g, b values of rgb2
        double b2 = (rgb2) & 0xff;
        double g2 = (rgb2 >> 8) & 0xff;
        double r2 = (rgb2 >> 16) & 0xff;

        double energy = (r1 - r2) * (r1 - r2) + (g1 - g2) * (g1 - g2) + (b1 - b2) * (b1 - b2);

        return energy;


    }

    /**
     * This method finds a horizontal seam with minimum energy in the given energy table.
     *
     * @param energyTable The energy table that will be used to find a horizontal seam.
     * @return A horizontal seam with minimum energy.
     */
    private Seam getHorizontalSeam(double[][] energyTable) {
        int width = energyTable.length;
        int height = energyTable[0].length;

        // initialize the seam that will be returned
        Seam seam = new Seam(width, "horizontal");

        // 2d array keeps the dynamic solution
        double[][] horizontal_dp = new double[width][height];

        // 2d array for backtracking
        int[][] prev = new int[width][height];

        // loop over all the pixels
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                double min_value;

                // base case
                if (i == 0) {
                    horizontal_dp[i][j] = energyTable[i][j];
                    prev[i][j] = -1;
                    continue;
                }

                // if on the edge, there are 2 pixels to take the minimum
                else if (j == 0) {
                    min_value = Math.min(horizontal_dp[i - 1][j], horizontal_dp[i - 1][j + 1]);
                    if (min_value == horizontal_dp[i - 1][j]) {
                        prev[i][j] = j;
                    } else {
                        prev[i][j] = j + 1;
                    }
                }
                // if on the edge, there are 2 pixels to take the minimum
                else if (j == height - 1) {
                    min_value = Math.min(horizontal_dp[i - 1][j], horizontal_dp[i - 1][j - 1]);
                    if (min_value == horizontal_dp[i - 1][j]) {
                        prev[i][j] = j;
                    } else {
                        prev[i][j] = j - 1;
                    }
                }
                // otherwise take the minimum of three neighbor pixels
                else {
                    min_value = Math.min(horizontal_dp[i - 1][j], Math.min(horizontal_dp[i - 1][j - 1], horizontal_dp[i - 1][j + 1]));

                    if (min_value == horizontal_dp[i - 1][j]) {
                        prev[i][j] = j;
                    } else if (min_value == horizontal_dp[i - 1][j - 1]) {
                        prev[i][j] = j - 1;
                    } else {
                        prev[i][j] = j + 1;
                    }


                }

                // add min value to the current energy
                horizontal_dp[i][j] = energyTable[i][j] + min_value;
            }
        }


        // find the minimum total energy on the edge
        // and its coordinate
        double min_energy = horizontal_dp[width - 1][0];
        int min_coord = 0;
        for (int j = 0; j < height; j++) {
            if (min_energy > horizontal_dp[width - 1][j]) {
                min_energy = horizontal_dp[width - 1][j];
                min_coord = j;
            }
        }

        seam.setEnergy(min_energy);

        // backtrack from the minimum, and build the seam
        for (int i = width - 1; i >= 0; i--) {
            seam.setPixels(i, min_coord);
            min_coord = prev[i][min_coord];
        }

        return seam;
    }

    /**
     * This method finds a vertical seam with minimum energy in the given energy table.
     *
     * @param energyTable The energy table that will be used to find a vertical seam.
     * @return A vertical seam with minimum energy.
     */
    private Seam getVerticalSeam(double[][] energyTable) {
        int width = energyTable.length;
        int height = energyTable[0].length;

        // initialize the seam that will be returned
        Seam seam = new Seam(height, "vertical");

        // 2d array keeps the dynamic solution
        double[][] vertical_dp = new double[width][height];

        // 2d array for backtracking
        int[][] prev = new int[width][height];

        // loop over all the pixels
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                double min_value;

                // base case
                if (j == 0) {
                    vertical_dp[i][j] = energyTable[i][j];
                    prev[i][j] = -1;
                    continue;
                }
                // if on the edge, there are 2 pixels to take the minimum
                else if (i == 0) {
                    min_value = Math.min(vertical_dp[i][j - 1], vertical_dp[i + 1][j - 1]);
                    if (min_value == vertical_dp[i][j - 1]) {
                        prev[i][j] = i;
                    } else {
                        prev[i][j] = i + 1;
                    }
                }
                // if on the edge, there are 2 pixels to take the minimum
                else if (i == width - 1) {
                    min_value = Math.min(vertical_dp[i][j - 1], vertical_dp[i - 1][j - 1]);
                    if (min_value == vertical_dp[i][j - 1]) {
                        prev[i][j] = i;
                    } else {
                        prev[i][j] = i - 1;
                    }
                }
                // otherwise take the minimum of three neighbor pixels
                else {
                    min_value = Math.min(vertical_dp[i][j - 1], Math.min(vertical_dp[i - 1][j - 1], vertical_dp[i + 1][j - 1]));

                    if (min_value == vertical_dp[i][j - 1]) {
                        prev[i][j] = i;
                    } else if (min_value == vertical_dp[i - 1][j - 1]) {
                        prev[i][j] = i - 1;
                    } else {
                        prev[i][j] = i + 1;
                    }


                }

                // add min value to the current energy
                vertical_dp[i][j] = energyTable[i][j] + min_value;
            }
        }


        // find the minimum total energy on the edge
        // and its coordinate
        double min_energy = vertical_dp[0][height - 1];
        int min_coord = 0;
        for (int i = 0; i < width; i++) {
            if (min_energy > vertical_dp[i][height - 1]) {
                min_energy = vertical_dp[i][height - 1];
                min_coord = i;
            }
        }

        seam.setEnergy(min_energy);

        // backtrack from the minimum, and build the seam
        for (int j = height - 1; j >= 0; j--) {
            seam.setPixels(j, min_coord);
            min_coord = prev[min_coord][j];
        }

        return seam;
    }

    /**
     * This method removes a given seam from the image.
     *
     * @param seam The seam that will be removed.
     */
    private void removeSeam(Seam seam) {
        int width = carvedImage.getWidth();
        int height = carvedImage.getHeight();


        if (seam.getDirection().equals("horizontal")) {
            Map<Point, Integer> newPixels = new HashMap<>();
            carvedImage.decrementHeight();
            // loop over all pixels
            for (int i = 0; i < width; i++) {
                boolean moveToNext = false;
                for (int j = 0; j < height - 1; j++) {
                    // once we run into the pixel in the seam
                    // skip it and keep copying from the next one
                    if (seam.getPixels()[i] == j) {
                        moveToNext = true;
                    }
                    if (moveToNext){
                        newPixels.put(new Point(i, j), carvedImage.getRGB(i, j + 1));
                    }

                }
            }
            newPixels.entrySet().forEach(p -> carvedImage.setRGB(p.getKey().x, p.getKey().y, p.getValue()));
        } else {
            // decrement the width by 1
            Map<Point, Integer> newPixels = new HashMap<>();
            carvedImage.decrementWidth();
            // loop over all pixels
            for (int j = 0; j < height; j++) {
                boolean moveToNext = false;
                for (int i = 0; i < width - 1; i++) {
                    // once we run into the pixel in the seam
                    // skip it and keep copying from the next one
                    if (seam.getPixels()[j] == i) {
                        moveToNext = true;
                    }
                    if (moveToNext) {
                        newPixels.put(new Point(i, j), carvedImage.getRGB(i + 1, j));
                    }
                }
            }
            newPixels.entrySet().forEach(p -> carvedImage.setRGB(p.getKey().x, p.getKey().y, p.getValue()));
        }

    }

}