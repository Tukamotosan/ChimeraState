#include "ChimeraSim.h"



ChimeraSim::ChimeraSim()
{
}


ChimeraSim::~ChimeraSim()
{
}

/*!
*/
void ChimeraSim::init(const float c1_, const float c2_, const float nu_, const float eta_, const int w_, const int h_, const float noise_) {
	c1 = c1_;
	c2 = c2_;
	nu = nu_;
	eta = eta_;
	w = w_;
	h = h_;
	noise = noise_;
	rng(getTickCount());
	dt = 0.05;
	t = 0.0;

	W_real = Mat::zeros(w, h, CV_32F);
	W_imag = Mat::zeros(w, h, CV_32F);
	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			W_real.at<float>(x, y) = (float)rng.gaussian(noise);
			W_imag.at<float>(x, y) = (float)rng.gaussian(noise);
		}
	}

	W_dash_real = Mat::zeros(w, h, CV_32F);
	W_dash_imag = Mat::zeros(w, h, CV_32F);

	LapFilter = Mat::zeros(3, 3, CV_32F);
	LapFilter.at<float>(0, 0) = 0.0;
	LapFilter.at<float>(0, 1) = 1.0;
	LapFilter.at<float>(0, 2) = 0.0;

	LapFilter.at<float>(1, 0) =  1.0;
	LapFilter.at<float>(1, 1) = -4.0;
	LapFilter.at<float>(1, 2) =  1.0;

	LapFilter.at<float>(2, 0) = 0.0;
	LapFilter.at<float>(2, 1) = 1.0;
	LapFilter.at<float>(2, 2) = 0.0;
}

void ChimeraSim::exec() {
	// calculate eq.(1)

	// 1st term
	Mat A_real = W_real;
	Mat A_imag = W_imag;

	// 2nd term
	Mat B_real, B_imag;
	filter2D(W_real, B_real, -1, LapFilter);
	filter2D(W_imag, B_imag, -1, LapFilter);
	Mat B_real2 = B_real - c1*B_imag;
	Mat B_imag2 = c1*B_real + B_imag;

	// 3,4,5th term
	vector<Mat> XS = f_3to5th_term(W_real, W_imag, c1, c2, nu, eta, t);

	W_dash_real = A_real + B_real - XS[0] - XS[2] + XS[4];
	W_dash_imag = A_real + B_real - XS[1] - XS[3] + XS[4];

	W_real += dt*W_dash_real;
	W_imag += dt*W_dash_imag;

	t += dt;
}

/*!
	path ‚ÉW_real‚Ì‰æ‘œ‚ğ•Û‘¶‚·‚é
*/
void ChimeraSim::save_w_real(const string path) {
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(W_real, &minVal, &maxVal, &minLoc, &maxLoc);

	Mat gray_image(h, w, CV_8UC1);
	float v;

	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			v = map(W_real.at<float>(x, y), minVal, maxVal, 0, 255);
			gray_image.at<unsigned char>(y, x) = (unsigned char)v;
		}
	}
	imwrite(path, gray_image);

}

/*!
path ‚ÉW_imag‚Ì‰æ‘œ‚ğ•Û‘¶‚·‚é
*/
void ChimeraSim::save_w_imag(const string path) {
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(W_imag, &minVal, &maxVal, &minLoc, &maxLoc);

	Mat gray_image(w, h, CV_8UC1);
	float v;

	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			v = map(W_imag.at<float>(x, y), minVal, maxVal, 0, 255);
			gray_image.at<unsigned char>(y, x) = (unsigned char)v;
		}
	}
	imwrite(path, gray_image);
}

void ChimeraSim::save_txt(const string path) {
	ofstream f_out(path);

	f_out << h << "\n";
	f_out << w << "\n";
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			f_out << std::to_string(W_real.at<float>(x, y)) << ",";
		}
		f_out << "\n";
	}
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			f_out << std::to_string(W_imag.at<float>(x, y)) << ",";
		}
		f_out << "\n";
	}

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			f_out << std::to_string(W_dash_real.at<float>(x, y)) << ",";
		}
		f_out << "\n";
	}

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			f_out << std::to_string(W_dash_imag.at<float>(x, y)) << ",";
		}
		f_out << "\n";
	}

	f_out.close();
}

/*!
®1‚Ì‘æ3,4,5€‚ğ•Ô‚·
*/
vector<Mat> ChimeraSim::f_3to5th_term(const Mat W_real, const Mat W_imag, const float c1, const float c2, const float nu, const float eta, const float t) {
	MatSize size = W_real.size;
	Mat X_real3 = Mat::zeros(size[0], size[1], CV_32F);
	Mat X_imag3 = Mat::zeros(size[0], size[1], CV_32F);
	Mat X_real4 = Mat::zeros(size[0], size[1], CV_32F);
	Mat X_imag4 = Mat::zeros(size[0], size[1], CV_32F);
	Mat X_real5 = Mat::zeros(size[0], size[1], CV_32F);
	Mat X_imag5 = Mat::zeros(size[0], size[1], CV_32F);

#pragma omp parallel for
	for (int x = 0; x < size[0]; x++) {
#pragma omp parallel for
		for (int y = 0; y < size[1]; y++) {
			float w_real = W_real.at<float>(x, y);
			float w_imag = W_imag.at<float>(x, y);
			float w_norm = norm2(w_real, w_imag);

			// ‘æ3€‚ÌŒvZ
			float t_real = w_norm * w_real;
			float t_imag = w_norm * w_imag;
			X_real3.at<float>(x, y) =    t_real - c2*t_imag;
			X_imag3.at<float>(x, y) = c2*t_real +    t_imag;

			// ‘æ4€‚ÌŒvZ
			t_real = eta*cos(nu*t);
			t_imag = -eta*sin(nu*t);
			X_real4.at<float>(x, y) = t_real - nu*t_imag;
			X_imag4.at<float>(x, y) = nu*t_real + t_imag;

			// ‘æ5€‚ÌŒvZ
			t_real = eta*w_norm*cos(nu*t);
			t_imag = -eta*w_norm*sin(nu*t);
			X_real5.at<float>(x, y) = t_real - c2*t_imag;
			X_imag5.at<float>(x, y) = c2*t_real + t_imag;
		}
	}
	vector<Mat> Y(6);
	Y[0] = X_real3;
	Y[1] = X_imag3;

	Y[2] = X_real4;
	Y[3] = X_imag4;

	Y[4] = X_real5;
	Y[5] = X_imag5;

	return Y;
}

/*!
®1‚Ì‘æO€‚ğ•Ô‚·
*/
vector<Mat> ChimeraSim::f_3rd_term(const Mat W_norm, const Mat W_real, const Mat W_imag, const float c2) {
	MatSize size = W_real.size;
	Mat X_real = Mat::zeros(size[0], size[1], CV_32F);
	Mat X_imag = Mat::zeros(size[0], size[1], CV_32F);

#pragma omp parallel for
	for (int x = 0; x < size[0]; x++) {
#pragma omp parallel for
		for (int y = 0; y < size[1]; y++) {
			X_real.at<float>(x, y) = W_norm.at<float>(x, y) * W_real.at<float>(x, y);
			X_imag.at<float>(x, y) = W_norm.at<float>(x, y) * W_imag.at<float>(x, y);
		}
	}

	vector<Mat> Y(2);
	Y[0] =    X_real - c2*X_imag;
	Y[1] = c2*X_real +    X_imag;
	return Y;
}

/*!
®1‚Ì‘æ4€‚ğ•Ô‚·
*/
vector<Mat> ChimeraSim::f_4th_term(const Mat W_real, const Mat W_imag, const float eta, const float nu, const float t) {
	MatSize size = W_real.size;
	Mat X_real = Mat::zeros(size[0], size[1], CV_32F);
	Mat X_imag = Mat::zeros(size[0], size[1], CV_32F);

#pragma omp parallel for
	for (int x = 0; x < size[0]; x++) {
#pragma omp parallel for
		for (int y = 0; y < size[1]; y++) {
			X_real.at<float>(x, y) = eta*cos(nu*t);
			X_imag.at<float>(x, y) = -eta*sin(nu*t);
		}
	}

	vector<Mat> Y(2);
	Y[0] = X_real;
	Y[1] = X_imag;
	return Y;
}

/*!
s—ñW‚ÌŠe—v‘f‚Ì‘å‚«‚³‚ğ—v‘f‚É‚µ‚½s—ñ‚ğ•Ô‚·
*/
Mat ChimeraSim::Wnorm2(const Mat W_real, const Mat W_imag) {
	MatSize size = W_real.size;
	Mat Y = Mat::zeros(size[0], size[1], CV_32F);

#pragma omp parallel for
	for (int x = 0; x < size[0]; x++) {
#pragma omp parallel for
		for (int y = 0; y < size[1]; y++) {
			Y.at<float>(x, y) = norm2(W_real.at<float>(x, y), W_imag.at<float>(x, y));
		}
	}

	return Y;
}

/*!
•¡‘f”‚Ì‘å‚«‚³‚Ì2æ‚ğ•Ô‚·
*/
float ChimeraSim::norm2(const float x, const float y) {
	return x*x + y*y;
}

float ChimeraSim::map(const float x, const float xMin, const float xMax, const float yMin, const float yMax) {
	return (x - xMin)*(yMax - yMin) / (xMax - xMin) + yMin;
}