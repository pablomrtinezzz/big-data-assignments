package src.java;

import java.util.Arrays;
import java.util.Locale;
import common.PRNG;

public class MatrixMultiplication {

    /**
     * Standard canonical triple-loop dense matrix multiplication (i-j-k).
     * Storage layout: double[][] (array of row references).
     * Arithmetic work: exactly 2 * n^3 operations (n^3 multiplications, n^3 additions).
     */
    public static void matmulIJK(double[][] A, double[][] B, double[][] C, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }

    public static double computeChecksum(double[][] C, int n) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sum += C[i][j];
            }
        }
        return sum;
    }

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);

        int n = 64;
        long seed = 42L;
        int warmups = 0;
        int repetitions = 1;
        boolean outputJson = false;
        boolean printMatrices = false;
        int testMode = 0; // 0: random, 1: identity, 2: zero

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--n":
                    if (i + 1 < args.length) n = Integer.parseInt(args[++i]);
                    break;
                case "--seed":
                    if (i + 1 < args.length) seed = Long.parseLong(args[++i]);
                    break;
                case "--warmups":
                    if (i + 1 < args.length) warmups = Integer.parseInt(args[++i]);
                    break;
                case "--repetitions":
                    if (i + 1 < args.length) repetitions = Integer.parseInt(args[++i]);
                    break;
                case "--json":
                    outputJson = true;
                    break;
                case "--print":
                    printMatrices = true;
                    break;
                case "--test-identity":
                    testMode = 1;
                    break;
                case "--test-zero":
                    testMode = 2;
                    break;
            }
        }

        if (n <= 0 || repetitions <= 0) {
            System.err.printf("Invalid parameters: n=%d, repetitions=%d%n", n, repetitions);
            System.exit(1);
        }

        System.gc();
        long memBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

        double[][] A = new double[n][n];
        double[][] B = new double[n][n];
        double[][] C = new double[n][n];

        PRNG prng = new PRNG(seed);
        if (testMode == 1) { // Identity: A random, B = Identity
            prng.fillMatrix(A, n);
            for (int i = 0; i < n; i++) B[i][i] = 1.0;
        } else if (testMode == 2) { // Zero: A random, B = Zero
            prng.fillMatrix(A, n);
        } else { // Standard random
            prng.fillMatrix(A, n);
            prng.fillMatrix(B, n);
        }

        long memAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        long heapUsed = Math.max(0, memAfter - memBefore);
        long totalPayloadBytes = 3L * n * n * Double.BYTES;

        // Warm-up runs (not timed/reported)
        for (int w = 0; w < warmups; w++) {
            matmulIJK(A, B, C, n);
        }

        // Timed repetitions
        double[] timesSec = new double[repetitions];
        double lastChecksum = 0.0;

        for (int r = 0; r < repetitions; r++) {
            // Reset output matrix
            for (int i = 0; i < n; i++) {
                Arrays.fill(C[i], 0.0);
            }

            long t0 = System.nanoTime();
            matmulIJK(A, B, C, n);
            long t1 = System.nanoTime();

            timesSec[r] = (t1 - t0) * 1e-9;
            lastChecksum = computeChecksum(C, n);
        }

        // Compute statistics
        double[] sortedTimes = Arrays.copyOf(timesSec, repetitions);
        Arrays.sort(sortedTimes);

        double medianTime;
        if (repetitions % 2 == 1) {
            medianTime = sortedTimes[repetitions / 2];
        } else {
            medianTime = 0.5 * (sortedTimes[repetitions / 2 - 1] + sortedTimes[repetitions / 2]);
        }

        double q1, q3, iqr;
        if (repetitions == 1) {
            q1 = sortedTimes[0];
            q3 = sortedTimes[0];
            iqr = 0.0;
        } else {
            int mid = repetitions / 2;
            if (mid % 2 == 1) {
                q1 = sortedTimes[mid / 2];
                q3 = sortedTimes[repetitions - 1 - (mid / 2)];
            } else {
                q1 = 0.5 * (sortedTimes[mid / 2 - 1] + sortedTimes[mid / 2]);
                q3 = 0.5 * (sortedTimes[repetitions - mid + mid / 2 - 1] + sortedTimes[repetitions - mid + mid / 2]);
            }
            iqr = q3 - q1;
        }

        double minTime = sortedTimes[0];
        double maxTime = sortedTimes[repetitions - 1];

        if (printMatrices && n <= 10) {
            System.out.printf("Result matrix C (%dx%d):%n", n, n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    System.out.printf("%10.6f ", C[i][j]);
                }
                System.out.println();
            }
        }

        if (outputJson) {
            System.out.println("{");
            System.out.println("  \"language\": \"java\",");
            System.out.printf("  \"n\": %d,%n", n);
            System.out.printf("  \"seed\": %d,%n", seed);
            System.out.printf("  \"warmups\": %d,%n", warmups);
            System.out.printf("  \"repetitions\": %d,%n", repetitions);
            System.out.print("  \"times_sec\": [");
            for (int r = 0; r < repetitions; r++) {
                System.out.printf(Locale.US, "%.9f%s", timesSec[r], (r == repetitions - 1) ? "" : ", ");
            }
            System.out.println("],");
            System.out.printf(Locale.US, "  \"median_time_sec\": %.9f,%n", medianTime);
            System.out.printf(Locale.US, "  \"iqr_sec\": %.9f,%n", iqr);
            System.out.printf(Locale.US, "  \"min_time_sec\": %.9f,%n", minTime);
            System.out.printf(Locale.US, "  \"max_time_sec\": %.9f,%n", maxTime);
            System.out.printf("  \"payload_bytes\": %d,%n", totalPayloadBytes);
            System.out.printf("  \"heap_bytes\": %d,%n", heapUsed);
            System.out.printf(Locale.US, "  \"checksum\": %.12e%n", lastChecksum);
            System.out.println("}");
        } else {
            System.out.printf(Locale.US,
                "Java (n=%d): Repetitions=%d, Median=%.6f s, IQR=%.6f s, Min=%.6f s, Max=%.6f s, Heap=%.2f MB, Checksum=%.6f%n",
                n, repetitions, medianTime, iqr, minTime, maxTime, (double)heapUsed / (1024.0 * 1024.0), lastChecksum);
        }
    }
}
