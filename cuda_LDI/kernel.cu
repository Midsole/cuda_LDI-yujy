#define  _CRT_SECURE_NO_WARNINGS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// depthpeeling.cpp : 定义控制台应用程序的入口点。
//

#include <vector>
#include <string>
#include <thread>
#include <GL/glew.h>
#include <gl/GL.h>
#include <GLFW/glfw3.h>	// Window & keyboard, contains OpenGL
#include <omp.h>
#include <iostream>

#include <SOIL.h>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <Windows.h>
#include <cv.h>
#include <highgui.h>
#include <math.h>
#include <fstream>

using namespace std;

float seta;

void printfbf(glm::mat4 matrix) {
	(float*)glm::value_ptr(matrix);
	int counts = 0;
	for (size_t i = 0; i < 16; i++) {
		//	printf("%d", i % 4);
		printf("%f ", matrix[i / 4][i % 4]);
		counts++;
		if (counts % 4 == 0) {
			printf("\n");
		}
	}
}

void mat4toarray(glm::mat4 matrix, float* target_array) {
	(float*)glm::value_ptr(matrix);
	int counts = 0;
	for (size_t i = 0; i < 16; i++) {
		//	printf("%d", i % 4);
		target_array[i] = matrix[i / 4][i % 4];
	}
}

void printfarr(float* target_array){
	printf("the array is:");
	for (size_t i = 0; i < 16; i++)
	{
		printf(" %f,", target_array[i]);
	}
	printf("\n");
}

__global__ void calculate_next_pos(float* d_layer0, float* d_layer1, float* d_outputIMG, float* d_MVP,int Height, int Width, float* d_scaler) {
	float fi, fj;
	float stepsize = 0.001;
	fi = blockIdx.x*blockDim.x + threadIdx.x;
	fj = blockIdx.y*blockDim.y + threadIdx.y;
	int i, j;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	j = blockIdx.y*blockDim.y + threadIdx.y;
	//float fi = (float)i;
	//float fj = (float)j;
	//MVP*vec4(float(j), float(i), depth, 1.0f);
	//int new_y = floor(d_MVP[0] * j + d_MVP[1] * i + d_MVP[2] * d_layer0[i*1440 + j] + d_MVP[3]);
	//int new_x = floor(d_MVP[4] * j + d_MVP[5] * i + d_MVP[6] * d_layer0[i*1440 + j] + d_MVP[7]);

	float new_y = d_MVP[0] * fj + d_MVP[1] * fi + d_MVP[2] * 20 * d_layer1[i * 1440 + j] + d_MVP[3] + (j - Width / 2)*d_layer1[i * 1440 + j] * (*d_scaler)*stepsize;
	float new_x = d_MVP[4] * fj + d_MVP[5] * fi + d_MVP[6] * 20 * d_layer1[i * 1440 + j] + d_MVP[7] + (i - Height / 2)*d_layer1[i * 1440 + j] * (*d_scaler)*stepsize;

	int new_xi = (int)new_x;
	int new_yi = (int)new_y;
	if (new_x >= 0 && new_x < 768 && new_y >= 0 && new_y < 1440)
	{
		if (d_outputIMG[new_xi * 1440 + new_yi] < d_layer1[i * 1440 + j]/256)
		//if(true)
		{
			d_outputIMG[new_xi * 1440 + new_yi] = d_layer1[i * 1440 + j] / 256;//d_layer0[i*Width + j];
		}

	}
	/*
	new_y = d_MVP[0] * fj + d_MVP[1] * fi + d_MVP[2] * 20 * d_layer0[i * 1440 + j] + d_MVP[3] + (j - Width / 2)*d_layer0[i * 1440 + j] * (*d_scaler)*stepsize;
	new_x = d_MVP[4] * fj + d_MVP[5] * fi + d_MVP[6] * 20 * d_layer0[i * 1440 + j] + d_MVP[7] + (i - Height / 2)*d_layer0[i * 1440 + j] * (*d_scaler)*stepsize;
	new_xi = (int)new_x;
	new_yi = (int)new_y;
	if (new_x >= 0 && new_x < 768 && new_y >= 0 && new_y < 1440)
	{
		if (d_outputIMG[new_xi * 1440 + new_yi] < d_layer0[i * 1440 + j]/256)
		//if(true)
		{
			d_outputIMG[new_xi * 1440 + new_yi] = d_layer0[i * 1440 + j] / 256;//d_layer0[i*Width + j];
		}
		
	}
	*/

	//d_outputIMG[1440*i+j] = d_layer0[i * 1440 + j];
}

__global__ void calculate_next_pos1(float* d_layer0, float* d_layer1, float* d_outputIMG, float* d_MVP, int Height, int Width, float* d_scaler) {
	float fi, fj;
	float stepsize = 0.001;
	fi = blockIdx.x*blockDim.x + threadIdx.x;
	fj = blockIdx.y*blockDim.y + threadIdx.y;
	int i, j;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	j = blockIdx.y*blockDim.y + threadIdx.y;
	//float fi = (float)i;
	//float fj = (float)j;
	//MVP*vec4(float(j), float(i), depth, 1.0f);
	//int new_y = floor(d_MVP[0] * j + d_MVP[1] * i + d_MVP[2] * d_layer0[i*1440 + j] + d_MVP[3]);
	//int new_x = floor(d_MVP[4] * j + d_MVP[5] * i + d_MVP[6] * d_layer0[i*1440 + j] + d_MVP[7]);

	float new_y = d_MVP[0] * fj + d_MVP[1] * fi + d_MVP[2] * 20 * d_layer1[i * 1440 + j] + d_MVP[3] + (j - Width / 2)*d_layer1[i * 1440 + j] * (*d_scaler)*stepsize;
	float new_x = d_MVP[4] * fj + d_MVP[5] * fi + d_MVP[6] * 20 * d_layer1[i * 1440 + j] + d_MVP[7] + (i - Height / 2)*d_layer1[i * 1440 + j] * (*d_scaler)*stepsize;

	int new_xi = (int)new_x;
	int new_yi = (int)new_y;
	//draw the front layer

	new_y = d_MVP[0] * fj + d_MVP[1] * fi + d_MVP[2] * 20 * d_layer0[i * 1440 + j] + d_MVP[3] + (j - Width / 2)*d_layer0[i * 1440 + j] * (*d_scaler)*stepsize;
	new_x = d_MVP[4] * fj + d_MVP[5] * fi + d_MVP[6] * 20 * d_layer0[i * 1440 + j] + d_MVP[7] + (i - Height / 2)*d_layer0[i * 1440 + j] * (*d_scaler)*stepsize;
	new_xi = (int)new_x;
	new_yi = (int)new_y;
	if (new_x >= 0 && new_x < 768 && new_y >= 0 && new_y < 1440)
	{
		if (d_outputIMG[new_xi * 1440 + new_yi] < d_layer0[i * 1440 + j] / 256)
		//if(true)
		{
			d_outputIMG[new_xi * 1440 + new_yi] = d_layer0[i * 1440 + j] / 256;//d_layer0[i*Width + j];
		}

	}


	//d_outputIMG[1440*i+j] = d_layer0[i * 1440 + j];
}

void changeMVP(float*cpuMVP, float*gpuMVP,float seta,float beta,float gama,float*gpu_scaler) {

	glm::mat4 Pmatrix = glm::perspective(glm::radians(60.0f), 1.875f, 2.0f, 1350.0f);
	glm::mat4 eyePmatrix = glm::lookAt(glm::vec3(-16, -100, -72), glm::vec3(-24, -100, -108), glm::vec3(0, -1, 0));
	//printfbf(Pmatrix);
	//printfbf(eyePmatrix);
	glm::mat4 totalPmatrix = Pmatrix*eyePmatrix;
	glm::mat4 iPmatrix = glm::inverse(totalPmatrix);
	printf("iPmatrix\n");
	printfbf(iPmatrix);
	//Pmatrix = glm::perspective(glm::radians(60.0f), 1.875f, 2.0f, 1350.0f);
	//Pmatrix = glm::lookAt(glm::vec3(-16 + gama, -100, -72 + gama * 4.5), glm::vec3(-24 + seta +gama, -100 + beta, -108 + gama * 4.5), glm::vec3(0, -1, 0));
	glm::mat4 Cmatrix = glm::lookAt(glm::vec3(-16, -100, -72), glm::vec3(-24 + 4.5*seta/5, -100 + beta, -108-seta/5), glm::vec3(0, -1, 0));
	//Pmatrix = Pmatrix*eyePmatrix;
	/*
	float pi = 3.14159265;
	float rotatenum[16] = {
		1, 0, 0, 0,
		0, cos(seta*pi / 180), -sin(seta*pi / 180), 0,
		0, sin(seta*pi / 180), cos(seta*pi / 180), 0,
		0, 0, 0, 1
	};
	glm::mat4 rotate;
	memcpy(glm::value_ptr(rotate), rotatenum, sizeof(rotatenum));
	printfbf(rotate);
	*/
	float trans_x, trans_y, trans_z;
	trans_x = gama*0.5;
	trans_y = 0;
	trans_z = gama * 4.5*0.5;
	float transnum[16] = {
		1, 0, 0, trans_x,
		0, 1, 0, trans_y,
		0, 0, 1, trans_z,
		0, 0, 0, 1
	};
	glm::mat4 translate;
	memcpy(glm::value_ptr(translate), transnum, sizeof(transnum));


	
	glm::mat4 totalMVP = Pmatrix*Cmatrix*translate*iPmatrix;

	mat4toarray(totalMVP, cpuMVP);
	printf("cpuMVP = \n");
	printfarr(cpuMVP);

	cudaMemcpy(gpuMVP, cpuMVP, 16 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(gpu_scaler, &gama, sizeof(float), cudaMemcpyHostToDevice);
	


}



int main()
{
	clock_t startTime, endTime;
	startTime = clock();
	const int H = 667;
	const int W = 1000;
	float*dev_layer0;
	float*dev_layer1;
	float*dev_outputIMG;
	float*dev_MVP;
	float*dev_scaler;
	float scaler = 1;
	printf("start initializing data\n");
	/*
	float** layer0 = new float*[H];
	for (int i = 0; i < H; i++)
	{
		layer0[i] = new float[W];
	}
	float** layer1 = new float*[H];
	for (int i = 0; i < H; i++)
	{
		layer1[i] = new float[W];
	}
	float** outputIMG = new float*[H];
	for (int i = 0; i < H; i++)
	{
		outputIMG[i] = new float[W];
	}
	float** MVP = new float*[4];
	for (int i = 0; i < 4; i++)
	{
		MVP[i] = new float[4];
	}
	*/
	
	char* layer0;
	layer0 = (char *)malloc(sizeof(char) * W*H*3);
	float* layer1;
	layer1 = (float *)malloc(sizeof(float) * W*H);
	float* emptyIMG;
	emptyIMG = (float *)malloc(sizeof(float) * W*H);
	float* outputIMG;
	outputIMG = (float *)malloc(sizeof(float) * W*H);
	float* MVP;
	MVP = (float *)malloc(sizeof(float) * 4*4);
	
	//prepare for the MVP
	//glm::mat4 Pmatrix = glm::ortho<float>(-600, -100, -50, 250, 100, 600);
	glm::mat4 Pmatrix = glm::perspective(glm::radians(60.0f), 1.875f, 2.0f, 1350.0f);
	glm::mat4 eyePmatrix = glm::lookAt(glm::vec3(-16, -100, -72), glm::vec3(-24, -100, -108), glm::vec3(0, -1, 0));
	//printfbf(Pmatrix);
	//printf("eyePmatrix is:\n");
	//printfbf(eyePmatrix);
	Pmatrix = Pmatrix*eyePmatrix;
	glm::mat4 iPmatrix = glm::inverse(Pmatrix);
	//printfbf(iPmatrix);
	Pmatrix = glm::perspective(glm::radians(60.0f), 1.875f, 2.0f, 1350.0f)*glm::lookAt(glm::vec3(-16, -100, -72), glm::vec3(-24, -100, -108+1.0f), glm::vec3(0, -1, 0));


	/*
	seta = 20.0;
	float pi = 3.14159265;
	float rotatenum[16] = {
		1, 0, 0, 0,
		0, cos(seta*pi / 180), -sin(seta*pi / 180), 0,
		0, sin(seta*pi / 180), cos(seta*pi / 180), 0,
		0, 0, 0, 1
	};
	glm::mat4 rotate;
	memcpy(glm::value_ptr(rotate), rotatenum, sizeof(rotatenum));
	printfbf(rotate);
	*/
	//glm::mat4 totalMVP = Pmatrix*rotate*iPmatrix;
	glm::mat4 totalMVP = Pmatrix*iPmatrix;
	//printfbf(totalMVP);

	mat4toarray(totalMVP, MVP);
	//printf("MVP = \n");
	//printfarr(MVP);




	IplImage *img_layer0 = cvLoadImage("layer5.bmp", 1);
	IplImage *img_layer1 = cvLoadImage("perslayer0.bmp", 1);//next layer
	IplImage *output_img = cvLoadImage("perslayer0.bmp", 1);
	

	//int x, y;
	//for (y = 0; y<img_layer0->height; y++) {
	//	char *ptr = img_layer0->imageData + y * img_layer0->widthStep;
	//	for (x = 0; x<img_layer0->width; x++) {
	//		int temp = ptr[3 * x];
	//		ptr[3 * x] = ptr[3 * x + 1] = ptr[3 * x + 2] = temp; //这样就可以添加自己的操作，这里我使三通道颜色一样，就彩色图转黑白图了
	//	}
	//}


	ofstream fileout("imgtest.txt",ofstream::out);
	cout << img_layer0->width << "," << img_layer0->height << "," << img_layer0->widthStep;
	for (int y = 0; y<img_layer0->height; y++) {
		unsigned char* p = (unsigned char*)(img_layer0->imageData + y*img_layer0->widthStep);
		for (int x = 0; x<img_layer0->width*img_layer0->nChannels; x++)
			fileout << (int)p[x];
		fileout << '\n';
	}
	fileout.close();


	int step = img_layer0->widthStep;
	printf("step=%d",step);
	for (int i = 0; i < img_layer0->height; i++) {
		for (int j = 0; j < img_layer0->width; j++) {
			CvScalar s;
			s = cvGet2D(img_layer0, float(i), float(j)); // get the (i,j) pixel value in <float>
			layer0[i*3000+j *3]   = s.val[0];
			layer0[i*3000+j *3+1] = s.val[1];
			layer0[i*3000+j *3+2] = s.val[2];
			//layer0[i][j] = s.val[0];

			//////////////像素值改变赋值
		}
	}

	//layer0 = (int*)img_layer0->imageData;
	cout <<endl<<"size of img:  "<<(int)img_layer0->imageData[2001000-1]<<endl;
	img_layer0->imageData = (char*)layer0;
	cvNamedWindow("1");
	cvShowImage("1", img_layer0);
	cvWaitKey(0);





	/**
	//start CUDA initializing
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSethost failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	
	//copy imgs and matrixw
	cudaStatus = cudaMalloc((float**)&dev_layer0, W*H * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((float**)&dev_layer1, W*H * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((float**)&dev_outputIMG, W*H * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((float**)&dev_MVP, 4*4 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((float**)&dev_scaler, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	

	//copy data from DRAM to GPU
	int totalpixel = W*H;
	cudaStatus = cudaMemcpy(dev_MVP, MVP, 16 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy1 failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_layer0, layer0, totalpixel * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2 failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_layer1, layer1, totalpixel * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy3 failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_outputIMG, outputIMG, totalpixel * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy4 failed!");
		goto Error;
	}




	endTime = clock();
	cout << "initializing Time : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

	printf("start rendering\n");
	startTime = clock();
	dim3 block(4, 4, 1);
	dim3 grid(H / block.x, W / block.y, 1);
	calculate_next_pos << < grid, block, 0, 0 >> > (dev_layer0, dev_layer1, dev_outputIMG, dev_MVP, H, W,dev_scaler);

	printf("memory copy back\n");
	cudaStatus = cudaMemcpy(outputIMG, dev_outputIMG, totalpixel * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyback failed!");
		goto Error;
	}



	/*
#pragma omp parallel for
	for (int i = 0; i < img_layer0->height; i++) {
#pragma omp parallel for
		for (int j = 0; j < img_layer0->width; j++) {
			CvScalar s;
			s = cvGet2D(img_layer1, float(i), float(j)); // get the (i,j) pixel value in <float>
			s.val[0] = outputIMG[i * 1440 + j];
			s.val[1] = outputIMG[i * 1440 + j];
			s.val[2] = outputIMG[i * 1440 + j];
			cvSet2D(img_layer1, float(i), float(j), s);
			//////////////像素值改变赋值
		}
	}
	
	
	IplImage* output = cvCreateImage(cvSize(W,H), IPL_DEPTH_32F, 1);
	output->imageData = (char*)outputIMG;
	cvNamedWindow("1");
	cvShowImage("1", output);
	endTime = clock();
	cout << "Time per frame: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

	cvWaitKey(0);//一定要有 

	float testseta = 0.0f;
	float testbeta = 0.0f;
	float testgama = 0.0f;
	while (1) {
		test(&testseta, &testbeta, &testgama, MVP, dev_MVP, emptyIMG, dev_layer0, dev_layer1, output, outputIMG, dev_outputIMG, H, W, dev_scaler);

	}


Error:
	printf("\n%s\n", cudaGetErrorString(cudaStatus));
	cudaFree(dev_layer0);
	cudaFree(dev_layer1);
	cudaFree(dev_outputIMG);
	cudaFree(dev_MVP);
	free(layer0);
	free(layer1);
	free(outputIMG);
	free(MVP);
	Sleep(10000);
	return cudaStatus;
	*/
	return 0;
}


