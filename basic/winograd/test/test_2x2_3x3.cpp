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
		fake.push_back(rand());
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
/*
	conv3x3s1_winograd(
				wino_out.data(),
				cols,
			   	fakeImg.data(),
				cols,
				rows,
				cols,
				kernel);
*/
	return 0;
}
