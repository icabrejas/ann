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

	public Network(int[] sizes) {
		this.numLayers = sizes.length;
		this.sizes = sizes;
		this.biases = new RealMatrix[sizes.length - 1];
		this.weights = new RealMatrix[sizes.length - 1];
		for (int i = 1; i < sizes.length; i++) {
			this.biases[i - 1] = MatrixUtilities.randomMatrix(sizes[i], 1);
		}
		for (int i = 1; i < sizes.length; i++) {
			this.weights[i - 1] = MatrixUtilities.randomMatrix(sizes[i], sizes[i - 1]);
		}
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
		int n = trainingDataSet.size();
		for (int j = 0; j < epochs; j++) {
			Collections.shuffle(trainingDataSet);

			List<List<double[][]>> miniBatches = new ArrayList<>();
			for (int k = 0; k < n; k += miniBatchSize) {
				miniBatches.add(trainingDataSet.subList(k, k + miniBatchSize));
			}

			for (List<double[][]> miniBatch : miniBatches) {
				updateMiniBatch(miniBatch, eta);
			}
		}
	}

	private void updateMiniBatch(List<double[][]> miniBatch, double eta) {
		RealMatrix[] nablaB = new RealMatrix[sizes.length - 1];
		RealMatrix[] nablaW = new RealMatrix[sizes.length - 1];

		for (int i = 1; i < sizes.length; i++) {
			nablaB[i - 1] = MatrixUtils.createRealMatrix(sizes[i], 1);
		}
		for (int i = 1; i < sizes.length; i++) {
			nablaW[i - 1] = MatrixUtils.createRealMatrix(sizes[i], sizes[i - 1]);
		}

		for (double[][] sample : miniBatch) {
			double[] x = sample[0];
			double[] y = sample[1];

			RealMatrix[][] deltaNabla = backprop(x, y);

			RealMatrix[] deltaNablaB = deltaNabla[0];
			RealMatrix[] deltaNablaW = deltaNabla[1];

			for (int i = 0; i < nablaB.length; i++) {
				nablaB[i] = nablaB[i].add(deltaNablaB[i]);
				nablaW[i] = nablaW[i].add(deltaNablaW[i]);
			}
		}

		for (int i = 0; i < nablaB.length; i++) {
			biases[i] = biases[i].subtract(nablaB[i].scalarMultiply(eta / miniBatch.size()));
			weights[i] = weights[i].subtract(nablaW[i].scalarMultiply(eta / miniBatch.size()));
		}
	}

	private RealMatrix[][] backprop(double[] x, double[] y) {
		RealMatrix[] nablaB = new RealMatrix[sizes.length - 1];
		RealMatrix[] nablaW = new RealMatrix[sizes.length - 1];

		for (int i = 1; i < sizes.length; i++) {
			nablaB[i - 1] = MatrixUtils.createRealMatrix(sizes[i], 1);
		}
		for (int i = 1; i < sizes.length; i++) {
			nablaW[i - 1] = MatrixUtils.createRealMatrix(sizes[i], sizes[i - 1]);
		}

		RealMatrix activation = MatrixUtils.createColumnRealMatrix(x);
		List<RealMatrix> activations = new ArrayList<>(Arrays.asList(activation));
		List<RealMatrix> zs = new ArrayList<>();
		RealMatrix z;
		for (int i = 0; i < nablaB.length; i++) {
			z = weights[i].multiply(activation)
					.add(biases[i]);
			zs.add(z);
			activation = sigmoid(z);
			activations.add(activation);
		}

		RealMatrix A = costDerivative(activations.get(activations.size() - 1), MatrixUtils.createColumnRealMatrix(y));
		RealMatrix B = sigmoidPrime(zs.get(zs.size() - 1));
		RealMatrix delta = MatrixUtilities.apply(A, B, (a, b) -> a * b);
		
		nablaB[nablaB.length - 1] = delta;
		nablaW[nablaW.length - 1] = delta.multiply(activations.get(activations.size() - 2)
				.transpose());

		for (int l = 2; l < numLayers; l++) {
			z = zs.get(zs.size() - l);
			RealMatrix sp = sigmoidPrime(z);
			A = weights[weights.length - l + 1].transpose()
					.multiply(delta);
			delta = MatrixUtilities.apply(A, sp, (a, b) -> a * b);
			nablaB[nablaB.length - l] = delta;
			nablaW[nablaW.length - l] = delta.multiply(activations.get(activations.size() - l - 1)
					.transpose());
		}

		return new RealMatrix[][] { nablaB, nablaW };
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
