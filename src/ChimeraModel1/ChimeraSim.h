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
	dtだけ実行する
	*/
	void exec();

	/*!
	path にW_realの画像を保存する
	*/
	void save_w_real(const string path);

	/*!
	path にW_imagの画像を保存する
	*/
	void save_w_imag(const string path);

	void save_txt(const string path);

	/*!
	 複素数の大きさの2乗を返す
	*/
	float norm2(const float x, const float y);

	/*!
	行列Wの各要素の大きさを要素にした行列を返す
	*/
	Mat Wnorm2(const Mat W_real, const Mat W_imag);

	/*!
	式1の第3,4,5項を返す
	*/
	vector<Mat> f_3to5th_term(const Mat W_real, const Mat W_imag, const float c1, const float c2, const float nu, const float eta, const float t);

	/*!
	式1の第三項を返す
	*/
	vector<Mat> f_3rd_term(const Mat W_norm, const Mat W_real, const Mat W_imag, const float c2);

	/*!
	 式1の第4項を返す
	*/
	vector<Mat> f_4th_term(const Mat W_real, const Mat W_imag, const float eta, const float nu, const float t);

private:
	float map(const float x, const float xMin, const float xMax, const float yMin, const float yMax);
};

