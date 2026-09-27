/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   matrix.hpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: ldesboui <ldesboui@42angouleme.fr>         +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/09/04 10:39:14 by ldesboui          #+#    #+#             */
/*   Updated: 2026/09/27 15:43:04 by ldesboui         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#include <iostream>
#include <vector>
#include <initializer_list>
#include <cmath>
#include <string>

enum e_types
{
	NO_TYPE,
	SCALE,
	TRANSLATION,
	ROTATION,
	EMPTY
};

template <typename T>
class Matrix
{
	private:
		int				rows;
		int				cols;
		std::vector<T>	data;
		Matrix();
		template <typename U> friend class Matrix;
	public:
		/*
		 * -----------CONSTRUCTORS-------------
		 */
		//creates an identity Matrix
		Matrix(int cols)
		{
			if (cols <= 0)
				throw Matrix::TheException("Cannot instantiate a 0 rows/cols matrix");
			this->rows = cols;
			this->cols = cols;
			for (int i = 0; i < rows; ++i)
			{
				for (int y = 0; y < cols; ++y)
				{
					if (y == i)
						this->data.push_back(static_cast<T>(1));
					else
						this->data.push_back(static_cast<T>(0));
				}
			}
		}
		/*
		 * you can use this as Matrix(4, 4, SCALE, {sX, sY, sZ}, -1, -1), to create a scaling matrix for 3d
		 * this also works with TRANSLATION and ROTATION
		 * and also like this Matrix(2, 2, NO_TYPE, {val0.0, val0.1, val1.0, val1.1}, -1, -1) to get a fully created 2d Matrix
		 * or Matrix(2, 2, EMPTY, {}), 2x2 matrix full of 0
		 * the last 2 arguments are to choose the rotation axis, x = 0, y = 1 and so on
		 * for rotation you should pass your angle, not sin or something. We do the calculus.
		 */
		Matrix(int rows, int cols, int type, std::initializer_list<T> list, int firstAxis = 0, int secondAxis = 0)
		{
			if (cols <= 0 || rows <= 0)
				throw Matrix::TheException("Cannot instantiate a 0 rows/cols matrix");

			this->rows = rows;
			this->cols = cols;
			this->data.reserve(rows * cols);
			const T* listData = list.begin();
			size_t listSize = list.size();

			switch (type)
			{
				case (NO_TYPE):
					if (listSize < static_cast<size_t>(rows * cols))
						throw Matrix::TheException("Not enough values in list for a full matrix");
					for (int i = 0; i < rows; ++i)
					{
						for (int y = 0; y < cols; ++y)
						{
							this->data.push_back(listData[i * cols + y]);
						}
					}
					break;
				case (TRANSLATION):
					if (rows != cols)
						throw Matrix::TheException("Translation matrix should be a square");
					if (listSize < static_cast<size_t>(rows - 1))
						throw Matrix::TheException("Not enough values in list for translation");
					for (int i = 0; i < rows; ++i)
					{
						for (int y = 0; y < cols; ++y)
						{
							if (y == i)
								this->data.push_back(static_cast<T>(1));
							else if (y == cols - 1 && i != rows - 1)
								this->data.push_back(static_cast<T>(listData[i]));
							else
								this->data.push_back(static_cast<T>(0));
						}
					}
					break;
				case (SCALE):
					if (rows != cols)
						throw Matrix::TheException("Scale matrix should be a square");
					if (listSize < static_cast<size_t>(rows - 1))
						throw Matrix::TheException("Not enough values in list for scale");
					for (int i = 0; i < rows; ++i)
					{
						for (int y = 0; y < cols; ++y)
						{
							if (y == i && y != rows - 1)
								this->data.push_back(static_cast<T>(listData[i]));
							else if (y == i && y == rows - 1)
								this->data.push_back(static_cast<T>(1));
							else
								this->data.push_back(static_cast<T>(0));
						}
					}
					break;
				case (ROTATION):
					{
						if (rows != cols)
							throw Matrix::TheException("Rotation matrix should be a square");
						if (rows < 2)
							throw Matrix::TheException("Rotation matrix should be at least 2x2");
						if (firstAxis == secondAxis)
							throw Matrix::TheException("Rotation matrix should have 2 different axis");
						if (listSize < 1)
							throw Matrix::TheException("Not enough values in list for rotation");
						int biggest = std::max(firstAxis, secondAxis);
						if (rows <= biggest)
							throw Matrix::TheException("Axes out of bounds");
						for (int i = 0; i < rows; ++i)
						{
							for (int y = 0; y < cols; ++y)
							{
								if ((firstAxis == y && firstAxis == i) || (secondAxis == y && secondAxis == i))
									this->data.push_back(static_cast<T>(std::cos(listData[0])));
								else if (firstAxis == i && secondAxis == y)
									this->data.push_back(static_cast<T>(-std::sin(listData[0])));
								else if (firstAxis == y && secondAxis == i)
									this->data.push_back(static_cast<T>(std::sin(listData[0])));
								else if (y == i)
									this->data.push_back(static_cast<T>(1));
								else
									this->data.push_back(static_cast<T>(0));
							}
						}
						break;
					}
				case (EMPTY):
					for (int i = 0; i < rows; ++i)
					{
						for (int y = 0; y < cols; ++y)
						{
							this->data.push_back(static_cast<T>(0));
						}
					}
					break;
			}
		}
		Matrix(const Matrix<T>& aMatrix): rows(aMatrix.getRows()), cols(aMatrix.getCols())
	{
		this->data = aMatrix.getData();
	}

		//convert a type U to a type T

		template <typename U>
			explicit Matrix(const Matrix<U>& other) : rows(other.getRows()), cols(other.getCols())
		{
			this->data.reserve(other.data.size());
			for (size_t i = 0; i < other.data.size(); ++i)
			{
				this->data.push_back(static_cast<T>(other.data[i]));
			}
		}

		/*
		 * ----------- ADD SUB AND MULTIPLY -----------------
		 */
		void			add(Matrix<T>& aMatrix)
		{
			if (aMatrix.getCols() != this->cols || aMatrix.getRows() != this->rows)
				throw Matrix::TheException("Cannot add different size matrices");
			for (int i = 0; i < this->rows; ++i)
			{
				for (int y = 0; y < this->cols; ++y)
				{
					this->data[i * cols + y] = this->data[i * cols + y] + aMatrix.getData()[i * cols + y];
				}
			}
		}

		void			add(T scalar)
		{
			for (int i = 0; i < this->rows; ++i)
			{
				for (int y = 0; y < this->cols; ++y)
				{
					this->data[i * cols + y] = this->data[i * cols + y] + scalar;
				}
			}
		}
		void			sub(Matrix<T>& aMatrix)
		{
			if (aMatrix.getCols() != this->cols || aMatrix.getRows() != this->rows)
				throw Matrix::TheException("Cannot add different size matrices");
			for (int i = 0; i < this->rows; ++i)
			{
				for (int y = 0; y < this->cols; ++y)
				{
					this->data[i * cols + y] = this->data[i * cols + y] - aMatrix.getData()[i * cols + y];
				}
			}
		}

		void			sub(T scalar)
		{
			for (int i = 0; i < this->rows; ++i)
			{
				for (int y = 0; y < this->cols; ++y)
				{
					this->data[i * cols + y] = this->data[i * cols + y] - scalar;
				}
			}
		}

		void multiply(Matrix<T>& other)
		{
			if (this->cols != other.rows)
				throw Matrix::TheException("Matrix multiplication requires cols of first matrix to match rows of second matrix");

			Matrix<T> result(this->rows, other.cols, EMPTY, {}, -1, -1);
			for (int i = 0; i < this->rows; ++i)
			{
				for (int j = 0; j < other.cols; ++j)
				{
					T sum = 0;
					for (int k = 0; k < this->cols; ++k)
					{
						sum += this->data[i * this->cols + k] * other.data[k * other.cols + j];
					}
					result.data[i * result.cols + j] = sum;
				}
			}
			this->cols = result.getCols();
			this->rows = result.getRows();
			this->data = result.getData();
		}

		void			multiply(T scalar)
		{
			for (int i = 0; i < this->rows; ++i)
			{
				for (int y = 0; y < this->cols; ++y)
				{
					this->data[i * cols + y] = this->data[i * cols + y] * scalar;
				}
			}
		}

		/*
		 * ----------------transpose det and dot -----------------
		 */
		Matrix<T>			transpose()
		{
			Matrix<T> result(this->cols, this->rows, EMPTY, {}, -1, -1);
			for (int i = 0; i < this->rows; ++i)
			{
				for (int j = 0; j < this->cols; ++j)
				{
					result.data[j * result.cols + i] = this->data[i * this->cols + j];
				}
			}
			return result;
		}

		T det() const
		{
			if (this->rows != this->cols)
				throw Matrix::TheException("Determinant can only be calculated for square matrices");

			if (this->rows == 1)
				return this->data[0];

			if (this->rows == 2)
				return this->data[0] * this->data[3] - this->data[1] * this->data[2];

			if (this->rows == 3)
			{
				T a = this->data[0], b = this->data[1], c = this->data[2];
				T d = this->data[3], e = this->data[4], f = this->data[5];
				T g = this->data[6], h = this->data[7], i = this->data[8];
				return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
			}
			T determinant = 0;
			int sign = 1;
			for (int y = 0; y < this->cols; ++y)
			{
				Matrix<T> minorMat = this->getMinor(0, y);
				determinant += sign * this->data[y] * minorMat.det();
				sign = -sign;
			}
			return determinant;
		}


		T	dot(const Matrix<T>& other)		const
		{
			if (this->cols != 1 || other.cols != 1 || this->rows != other.rows)
				throw Matrix::TheException("Dot product requires two column-vectors of same size");

			T sum = 0;
			for (int i = 0; i < this->rows; ++i)
				sum += this->data[i] * other.data[i];
			return sum;
		}

		Matrix<T>	cross(const Matrix<T>& other)		const
		{
			if (this->rows != 3 || this->cols != 1 || other.rows != 3 || other.cols != 1)
				throw Matrix::TheException("Cross product is only defined for 3D vectors (3x1)");

			Matrix<T> result(3, 1, EMPTY, {}, -1, -1);
			result.data[0] = this->data[1] * other.data[2] - this->data[2] * other.data[1]; // X = y1*z2 - z1*y2
			result.data[1] = this->data[2] * other.data[0] - this->data[0] * other.data[2]; // Y = z1*x2 - x1*z2
			result.data[2] = this->data[0] * other.data[1] - this->data[1] * other.data[0]; // Z = x1*y2 - y1*x2

			return result;

		}

		/*
		 * ---------------- Accessors ----------------------------
		 */
		Matrix<T> getMinor(int dropRow, int dropCol) const
		{
			Matrix<T> result(this->rows - 1, this->cols - 1, EMPTY, {});
			int destRow = 0;

			for (int i = 0; i < this->rows; ++i)
			{
				if (i == dropRow) continue;

				int destCol = 0;
				for (int y = 0; y < this->cols; ++y)
				{
					if (y == dropCol) continue;

					result.data[destRow * result.cols + destCol] = this->data[i * this->cols + y];
					destCol++;
				}
				destRow++;
			}
			return result;
		}

		int				getRows()	const {return rows;}
		int				getCols()	const {return cols;}
		int				size()		const {return this->data.size();}
		void			print()		const
		{
			for (int i = 0; i < this->rows; ++i)
			{
				std::cout << "| ";
				for (int j = 0; j < this->cols; ++j)
				{
					std::cout <<  this->data[i * cols + j] << " ";	
				}
				std::cout << "|" << std::endl;
			}
		}
		const std::vector<T>&	getData()	const {return data;}
		class TheException: public std::exception
	{
		private:
			std::string message;
		public:
			~TheException() throw(){}
			TheException(std::string message){this->message = message;}
			const char *what() const throw(){return message.c_str();} 
	};
};
