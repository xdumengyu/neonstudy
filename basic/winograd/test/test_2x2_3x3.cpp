#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "func.h"
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace chrono;
void constructFakeImg(vector<float> &fake, size_t rows, size_t cols)
{
	size_t nElement = rows * cols;
	fake.reserve(nElement);
	srand(time(NULL));
	for (size_t i = 0; i < nElement; ++i) 
	{
		fake.push_back(1);
	}
	return ;
}
int fakeTest()
{
	const int rows = 1024;
	const int cols = 768;
	float kernel[9] = {
		1,1,1,
		1,1,1,
		1,1,1
	};
	size_t nElement = rows * cols;
	vector<float> fakeImg;
	constructFakeImg(fakeImg, rows, cols);
	vector<float> wino_out(nElement);
	printf("input snap\n");
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			printf("\t%f",fakeImg[i * cols + j]);
		}
		printf("\n");
	}
	conv3x3s1_winograd(
				wino_out.data(),
				cols,
			   	fakeImg.data(),
				cols,
				rows,
				cols,
				kernel);
	printf("output snap\n");
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			printf("\t%f",wino_out[i * cols + j]);
		}
		printf("\n");
	}
	return 0;
}
int main(int argc, char **argv)
{
	Mat srcImage = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat srcFloat;
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	srcImage.convertTo(srcFloat, CV_32FC1);
	Mat dstFloat(srcImage.rows, srcImage.cols, CV_32FC1);
	const int ITER = 20;
	float ker[9] = {
			1,1,1,
			1,1,1,
			1,1,1
	};
	//normalize kernel
	float sum = std::accumulate(ker, ker + 9, 0.0);
	std::transform(ker, ker + 9, ker, [sum](float x) { return x / sum;});
	Mat kernel(3,3, CV_32FC1, ker);
	// cv func
	auto cv_start = system_clock::now();
	for (int iter = 0; iter < ITER; ++iter) {
			filter2D(srcFloat, dstFloat, srcFloat.depth(), kernel);
	}
	
	auto cv_end = system_clock::now();
	auto dur = duration_cast<milliseconds>(cv_end - cv_start);
	cout<<"cv version cost:\n";
	cout<<static_cast<double>(dur.count()) / ITER<<" ms\n";
	Mat dstImage;
	dstFloat.convertTo(dstImage, CV_8UC1);
	imwrite("cv_out.png", dstImage);
	// my neon 2x2_3x3
	auto neon_start = system_clock::now();
	for (int iter = 0; iter < ITER; ++iter) {
		conv3x3s1_winograd(
						(float *)(dstFloat.data),
						cols,
						(float *)(srcFloat.data),
						cols,
						rows,
						cols,
						ker
						);
		
	}
	auto neon_end = system_clock::now();
	dur = duration_cast<milliseconds>(neon_end - neon_start);
	cout<<"neon version cost:\n";
	cout<<static_cast<double>(dur.count()) / ITER<<" ms\n";
	dstFloat.convertTo(dstImage, CV_8UC1);
	imwrite("neon_out.png", dstImage);
	
	return 0;
}
