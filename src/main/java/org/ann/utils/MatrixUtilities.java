package org.ann.utils;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class MatrixUtilities {

	public static RealMatrix apply(RealMatrix A, DoubleUnaryOperator f) {
		double[][] data = A.getData();
		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < data[i].length; j++) {
				data[i][j] = f.applyAsDouble(data[i][j]);
			}
		}
		return MatrixUtils.createRealMatrix(data);
	}

	public static RealMatrix apply(RealMatrix A, RealMatrix B, DoubleBinaryOperator f) {
		double[][] data = new double[A.getRowDimension()][A.getColumnDimension()];
		double[][] dataA = A.getData();
		double[][] dataB = B.getData();
		for (int i = 0; i < dataA.length; i++) {
			for (int j = 0; j < dataA[i].length; j++) {
				data[i][j] = f.applyAsDouble(dataA[i][j], dataB[i][j]);
			}
		}
		return MatrixUtils.createRealMatrix(data);
	}

	
	public static RealMatrix randomMatrix(int rows, int columns) {
		NormalDistribution Z = new NormalDistribution(0, 1);

		RealMatrix randomMatrix = MatrixUtils.createRealMatrix(rows, columns);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				randomMatrix.setEntry(i, j, Z.sample());
			}
		}

		return randomMatrix;
	}
	
}
