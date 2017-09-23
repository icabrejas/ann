package org.ann;

import java.util.List;
import java.util.function.Function;

import org.ann.utils.MatrixUtilities;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.descriptive.moment.Mean;

public interface TrainTracker {

	void accept(Network ann, int epoch, int minibatch);

	public static TrainTracker dummy() {
		return (ann, epoch, minibatch) -> {
		};
	}

	public static TrainTracker period(TrainTracker tracker, int period) {
		return new TrainTracker() {
			private int k;

			@Override
			public void accept(Network ann, int epoch, int minibatch) {
				if (k++ % period == 0) {
					tracker.accept(ann, epoch, minibatch);
				}
			}
		};
	}

	public static TrainTracker rmse(List<double[][]> dataSet, int period) {
		return logger(TrainTracker.Logger.rmse(dataSet), period);
	}

	public static TrainTracker rmse(List<double[][]> dataSet) {
		return logger(TrainTracker.Logger.rmse(dataSet));
	}

	public static TrainTracker testError(List<double[][]> dataSet, int period) {
		return logger(TrainTracker.Logger.testError(dataSet), period);
	}

	public static TrainTracker testError(List<double[][]> dataSet) {
		return logger(TrainTracker.Logger.testError(dataSet));
	}

	public static TrainTracker logger(TrainTracker.Logger logger, int period) {
		return period(logger(logger), period);
	}

	public static TrainTracker logger(TrainTracker.Logger logger) {
		return (ann, epoch, minibatch) -> System.out.println(logger.log(ann, epoch, minibatch));
	}

	public static String rmse(Network ann, List<double[][]> dataSet) {
		EuclideanDistance distance = new EuclideanDistance();
		Mean mean = new Mean();
		for (double[][] dataPair : dataSet) {
			double[] x = dataPair[0];
			double[] y = dataPair[1];
			double[] y_ = ann.feedForward(x);
			mean.increment(Math.pow(distance.compute(y, y_), 2));
		}
		return String.format("%.4f", Math.sqrt(mean.getResult()));
	}

	public static String testError(Network ann, List<double[][]> dataSet) {
		Mean mean = new Mean();
		dataSet.parallelStream()
				.map((double[][] dataPair) -> {
					double[] x = dataPair[0];
					double[] y = dataPair[1];
					double[] y_ = ann.feedForward(x);
					return MatrixUtilities.argmax(y) != MatrixUtilities.argmax(y_) ? 1 : 0;
				})
				.forEach(mean::increment);
		return String.format("%.4f", Math.sqrt(mean.getResult()));
	}

	// public static String testError(Network ann, List<double[][]> dataSet) {
	// Mean mean = new Mean();
	// for (double[][] dataPair : dataSet) {
	// double[] x = dataPair[0];
	// double[] y = dataPair[1];
	// double[] y_ = ann.feedForward(x);
	// mean.increment(MatrixUtilities.argmax(y) != MatrixUtilities.argmax(y_) ?
	// 1 : 0);
	// }
	// return String.format("%.4f", Math.sqrt(mean.getResult()));
	// }

	public static interface Logger {

		String log(Network ann, int epoch, int minibatch);

		public static TrainTracker.Logger logger(Function<Network, String> evaluator) {
			return (ann, epoch, minibatch) -> String.format("%03d", epoch) + '.' + String.format("%04d", minibatch)
					+ ": " + evaluator.apply(ann);
		}

		public static TrainTracker.Logger rmse(List<double[][]> dataSet) {
			return logger(ann -> TrainTracker.rmse(ann, dataSet));
		}

		public static TrainTracker.Logger testError(List<double[][]> dataSet) {
			return logger(ann -> TrainTracker.testError(ann, dataSet));
		}

	}
}
