package org.ann;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.ann.utils.MatrixUtilities;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class Network {

	private int numLayers;
	private int[] sizes;
	private RealMatrix[] biases;
	private RealMatrix[] weights;
	public TrainTracker trainTracker = TrainTracker.dummy();

	private Network() {
	}

	public static Network basic(int... sizes) {
		Network network = new Network();
		network.numLayers = sizes.length;
		network.sizes = sizes;
		network.biases = new RealMatrix[sizes.length - 1];
		network.weights = new RealMatrix[sizes.length - 1];
		for (int i = 1; i < sizes.length; i++) {
			network.biases[i - 1] = MatrixUtilities.randomMatrix(sizes[i], 1);
		}
		for (int i = 1; i < sizes.length; i++) {
			network.weights[i - 1] = MatrixUtilities.randomMatrix(sizes[i], sizes[i - 1]);
		}
		return network;
	}

	public static Network nabla(int... sizes) {
		Network network = new Network();
		network.numLayers = sizes.length;
		network.sizes = sizes;
		network.biases = new RealMatrix[sizes.length - 1];
		network.weights = new RealMatrix[sizes.length - 1];
		for (int i = 1; i < network.sizes.length; i++) {
			network.biases[i - 1] = MatrixUtils.createRealMatrix(network.sizes[i], 1);
		}
		for (int i = 1; i < network.sizes.length; i++) {
			network.weights[i - 1] = MatrixUtils.createRealMatrix(network.sizes[i], network.sizes[i - 1]);
		}
		return network;
	}

	public int getNumLayers() {
		return numLayers;
	}

	public int[] getSizes() {
		return sizes;
	}

	public RealMatrix[] getBiases() {
		return biases;
	}

	public RealMatrix[] getWeights() {
		return weights;
	}

	public void SGD(List<double[][]> trainingDataSet, int epochs, int miniBatchSize, double eta) {
		if (trainingDataSet.size() % miniBatchSize != 0) {
			throw new Error();
		}
		trainingDataSet = new ArrayList<>(trainingDataSet);
		for (int j = 0; j < epochs; j++) {
			List<List<double[][]>> miniBatches = miniBatches(trainingDataSet, miniBatchSize);
			for (int miniBatch = 0; miniBatch < miniBatches.size(); miniBatch++) {
				updateMiniBatch(miniBatches.get(miniBatch), eta);
				trainTracker.accept(this, j, miniBatch);
			}
		}
	}

	private List<List<double[][]>> miniBatches(List<double[][]> trainingDataSet, int miniBatchSize) {
		Collections.shuffle(trainingDataSet);
		List<List<double[][]>> miniBatches = new ArrayList<>();
		for (int k = 0; k < trainingDataSet.size(); k += miniBatchSize) {
			miniBatches.add(trainingDataSet.subList(k, k + miniBatchSize));
		}
		return miniBatches;
	}

	private void updateMiniBatch(List<double[][]> miniBatch, double eta) {
		Network nabla = Network.nabla(sizes);
		miniBatch.parallelStream()
				.map(this::calculateDelta)
				.forEach(nabla::add);
		for (int i = 0; i < nabla.biases.length; i++) {
			biases[i] = biases[i].subtract(nabla.biases[i].scalarMultiply(eta / miniBatch.size()));
			weights[i] = weights[i].subtract(nabla.weights[i].scalarMultiply(eta / miniBatch.size()));
		}
	}

	private void add(Network delta) {
		for (int i = 0; i < biases.length; i++) {
			biases[i] = biases[i].add(delta.biases[i]);
			weights[i] = weights[i].add(delta.weights[i]);
		}
	}

	private Network calculateDelta(double[][] sample) {
		RealMatrix x = MatrixUtils.createColumnRealMatrix(sample[0]);
		RealMatrix y = MatrixUtils.createColumnRealMatrix(sample[1]);
		return backprop(x, y);
	}

	private Network backprop(RealMatrix x, RealMatrix y) {

		Network nabla = Network.nabla(sizes);

		RealMatrix activation = x;
		List<RealMatrix> activations = new ArrayList<>(Arrays.asList(activation));
		List<RealMatrix> zs = new ArrayList<>();
		RealMatrix z;
		for (int i = 0; i < nabla.biases.length; i++) {
			z = weights[i].multiply(activation)
					.add(biases[i]);
			zs.add(z);
			activation = sigmoid(z);
			activations.add(activation);
		}

		RealMatrix A = costDerivative(activations.get(activations.size() - 1), y);
		RealMatrix B = sigmoidPrime(zs.get(zs.size() - 1));
		RealMatrix delta = MatrixUtilities.apply(A, B, (a, b) -> a * b);

		nabla.biases[nabla.biases.length - 1] = delta;
		nabla.weights[nabla.weights.length - 1] = delta.multiply(activations.get(activations.size() - 2)
				.transpose());

		for (int l = 2; l < numLayers; l++) {
			z = zs.get(zs.size() - l);
			RealMatrix sp = sigmoidPrime(z);
			A = weights[weights.length - l + 1].transpose()
					.multiply(delta);
			delta = MatrixUtilities.apply(A, sp, (a, b) -> a * b);
			nabla.biases[nabla.biases.length - l] = delta;
			nabla.weights[nabla.weights.length - l] = delta.multiply(activations.get(activations.size() - l - 1)
					.transpose());
		}

		return nabla;
	}

	private RealMatrix costDerivative(RealMatrix outputActivations, RealMatrix y) {
		return outputActivations.subtract(y);
	}

	private RealMatrix sigmoidPrime(RealMatrix z) {
		return MatrixUtilities.apply(z, x -> sigmoid(x) * (1 - sigmoid(x)));
	}

	public double[] feedForward(double[] input) {
		RealMatrix a = MatrixUtils.createColumnRealMatrix(input);
		for (int i = 0; i < numLayers - 1; i++) {
			a = sigmoid(weights[i].multiply(a)
					.add(biases[i]));
		}
		return a.getColumn(0);
	}

	public RealMatrix sigmoid(RealMatrix z) {
		return MatrixUtilities.apply(z, this::sigmoid);
	}

	private double sigmoid(double z) {
		return 1.0 / (1.0 + Math.exp(-z));
	}

}
