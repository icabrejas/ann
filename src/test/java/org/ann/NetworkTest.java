package org.ann;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.junit.Test;

import junit.framework.TestCase;

public class NetworkTest extends TestCase {

	@Test
	public void testDimension() {
		int[] layers = { 2, 7, 5 };
		Network ann = new Network(layers);

		assertEquals(2, ann.getWeights().length);
		assertEquals(2, ann.getBiases().length);

		assertEquals(2, ann.getWeights()[0].getColumnDimension());
		assertEquals(7, ann.getWeights()[0].getRowDimension());
		assertEquals(7, ann.getWeights()[1].getColumnDimension());
		assertEquals(5, ann.getWeights()[1].getRowDimension());

		assertEquals(1, ann.getBiases()[0].getColumnDimension());
		assertEquals(7, ann.getBiases()[0].getRowDimension());
		assertEquals(1, ann.getBiases()[1].getColumnDimension());
		assertEquals(5, ann.getBiases()[1].getRowDimension());
	}

	@Test
	public void test_train_ann_1_1() {
		int[] layers = { 1, 1 };
		List<double[][]> trainingDataSet = getIdealANNData(layers);
		Network ann = train(layers, trainingDataSet);
		assertRMSE(trainingDataSet, ann);
	}

	@Test
	public void test_train_ann_1_1_1() {
		int[] layers = { 1, 1, 1 };
		List<double[][]> trainingDataSet = getIdealANNData(layers);
		Network ann = train(layers, trainingDataSet);
		assertRMSE(trainingDataSet, ann);
	}

	@Test
	public void test_train_ann_1_2_1() {
		int[] layers = { 1, 2, 1 };
		List<double[][]> trainingDataSet = getIdealANNData(layers);
		Network ann = train(layers, trainingDataSet);
		assertRMSE(trainingDataSet, ann);
	}

	@Test
	public void test_train_ann_1_7_1() {
		int[] layers = { 1, 7, 1 };
		List<double[][]> trainingDataSet = getIdealANNData(layers);
		Network ann = train(layers, trainingDataSet);
		assertRMSE(trainingDataSet, ann);
	}

	@Test
	public void testIdenty() {
		System.out.println("f(x) = x");
		int[] layers = { 1, 4, 1 };
		List<double[][]> trainingDataSet = getTrainingDataSet(x -> x, 1000);
		Network ann = train(layers, trainingDataSet);
		assertRMSE(trainingDataSet, ann);
	}

	@Test
	public void testSquare() {
		System.out.println("f(x) = x^2");
		int[] layers = { 1, 4, 1 };
		List<double[][]> trainingDataSet = getTrainingDataSet(x -> Math.pow(x, 2), 1000);
		Network ann = train(layers, trainingDataSet);
		assertRMSE(trainingDataSet, ann);
	}

	@Test
	public void testRoot() {
		System.out.println("f(x) = x^(1/2)");
		int[] layers = { 1, 4, 1 };
		List<double[][]> trainingDataSet = getTrainingDataSet(x -> Math.sqrt(x), 1000);
		Network ann = train(layers, trainingDataSet);
		assertRMSE(trainingDataSet, ann);
	}

	@Test
	public void testExp() {
		System.out.println("f(x) = e^(-x)");
		int[] layers = { 1, 4, 1 };
		List<double[][]> trainingDataSet = getTrainingDataSet(x -> Math.exp(-x), 1000);
		Network ann = train(layers, trainingDataSet);
		assertRMSE(trainingDataSet, ann);
	}

	@Test
	public void testParaboloid() {
		System.out.println("f(x, y) = (x^2 + y^2)/2");
		int[] layers = { 2, 7, 1 };
		List<double[][]> trainingDataSet = getTrainingDataSet(x -> (x[0] * x[0] + x[1] * x[1]) / 2, layers[0], 1000);
		Network ann = train(layers, trainingDataSet);
		assertRMSE(trainingDataSet, ann);
	}

	private void assertRMSE(List<double[][]> trainingDataSet, Network ann) {
		assertTrue("" + rmse(ann, trainingDataSet), rmse(ann, trainingDataSet) < 5E-2);
	}

	private List<double[][]> getIdealANNData(int[] layers) {
		Network idealANN = new Network(layers);
		return getTrainingDataSet(idealANN::feedForward, layers[0], layers[layers.length - 1], 1000);
	}

	private List<double[][]> getTrainingDataSet(DoubleUnaryOperator funct, int size) {
		return getTrainingDataSet(x -> new double[] { funct.applyAsDouble(x[0]) }, 1, 1, size);
	}

	private List<double[][]> getTrainingDataSet(Function<double[], Double> funct, int in, int size) {
		return getTrainingDataSet(x -> new double[] { funct.apply(x) }, in, 1, size);
	}

	private List<double[][]> getTrainingDataSet(Function<double[], double[]> funct, int in, int out, int size) {
		List<double[][]> dataset = new ArrayList<>();
		for (int i = 0; i < size; i++) {
			double[] x = random(in);
			dataset.add(new double[][] { x, funct.apply(x) });
		}
		return dataset;
	}

	private double[] random(int length) {
		double[] x = new double[length];
		for (int j = 0; j < x.length; j++) {
			x[j] = Math.random();
		}
		return x;
	}

	private Network train(int[] layers, List<double[][]> trainingDataSet) {
		System.out.println(Arrays.toString(layers));
		Network ann = new Network(layers);
		ann.trainTracker = TrainTracker.rmse(trainingDataSet, 100);
		int epochs = 25 * (layers[0] * layers[0]);
		ann.SGD(trainingDataSet, epochs, trainingDataSet.size() / 50, 10.0);
		System.out.println();
		return ann;
	}

	private static double rmse(Network ann, List<double[][]> trainingDataSet) {
		Mean mean = new Mean();
		for (double[][] dataPair : trainingDataSet) {
			double[] x = dataPair[0];
			double[] y = dataPair[1];
			mean.increment(Math.pow(y[0] - ann.feedForward(x)[0], 2));
		}
		return Math.sqrt(mean.getResult());
	}
}
