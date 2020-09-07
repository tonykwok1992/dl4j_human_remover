package removehuman.seamcarving;

class ProgressRecorder {

    private int lastNonZeroCount = Integer.MAX_VALUE;
    private int noImproveCount = 0;
    private final int threshold;

    public ProgressRecorder(int threshold) {
        this.threshold = threshold;
    }

    public boolean record(int currentNonZeroCount) {
        if (lastNonZeroCount == currentNonZeroCount) {
            noImproveCount++;
        } else {
            noImproveCount = 0;
        }
        lastNonZeroCount = currentNonZeroCount;
        return noImproveCount < threshold;
    }

}