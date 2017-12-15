#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "func.h"
#include <vector>
using namespace std;
void constructFakeImg(vector<float> &fake, size_t rows, size_t cols)
{
	int nElement = rows * cols;
	fake.reserve(nElement);
	srand(time(NULL));
	for (size_t i = 0; i < nElement; ++i) 
	{
		fake.push_back(1);
	}
	return ;
}
int main()
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
