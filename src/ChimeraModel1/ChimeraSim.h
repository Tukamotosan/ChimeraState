#pragma once
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string>
#include<iostream>
#include<fstream>

using namespace cv;
using namespace std;

class ChimeraSim
{
public:
	float dt, t;
	int w, h;
	float noise;
	float c1, c2, nu, eta;
	Mat W_real, W_imag;
	
	Mat W_dash_real, W_dash_imag;

	Mat LapFilter;
	// generator of norm
	RNG rng;

	ChimeraSim();
	~ChimeraSim();

	/*!
	*/
	void init(const float c1, const float c2, const float nu, const float eta, const int w, const int h, const float noise);

	/*!
	dt�������s����
	*/
	void exec();

	/*!
	path ��W_real�̉摜��ۑ�����
	*/
	void save_w_real(const string path);

	/*!
	path ��W_imag�̉摜��ۑ�����
	*/
	void save_w_imag(const string path);

	void save_txt(const string path);

	/*!
	 ���f���̑傫����2���Ԃ�
	*/
	float norm2(const float x, const float y);

	/*!
	�s��W�̊e�v�f�̑傫����v�f�ɂ����s���Ԃ�
	*/
	Mat Wnorm2(const Mat W_real, const Mat W_imag);

	/*!
	��1�̑�3,4,5����Ԃ�
	*/
	vector<Mat> f_3to5th_term(const Mat W_real, const Mat W_imag, const float c1, const float c2, const float nu, const float eta, const float t);

	/*!
	��1�̑�O����Ԃ�
	*/
	vector<Mat> f_3rd_term(const Mat W_norm, const Mat W_real, const Mat W_imag, const float c2);

	/*!
	 ��1�̑�4����Ԃ�
	*/
	vector<Mat> f_4th_term(const Mat W_real, const Mat W_imag, const float eta, const float nu, const float t);

private:
	float map(const float x, const float xMin, const float xMax, const float yMin, const float yMax);
};

