#include <opencv2/photo.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <iomanip>      // std::setprecision
#include <fstream>
using namespace cv;
using namespace std;

struct SHist{
	vector<Point_<short>> _vBXY[256];
	vector<Point_<short>> _vGXY[256];
	vector<Point_<short>> _vRXY[256];
};

void loadExposureSeq(vector<Mat>&, vector<float>&, vector<short>&, vector<vector<Mat>>&, vector<SHist>&);

void storeCRF(Mat& invCrf_){
	FileStorage fs("..//crf.yml", FileStorage::WRITE);
	fs << "inv_Cam_Response_Func" << invCrf_;
	return;
}
void loadCRF(Mat* pInvCrf_){
	FileStorage fs("..//crf.yml", FileStorage::READ);
	fs["inv_Cam_Response_Func"] >> *pInvCrf_;
	return;
}

void mean(short ImgCounter, const vector<Mat>& vImg, const SHist& hist, vector<Mat>& vBlue)
{
	//vIntensity[256] ImgCounter x # of pixel at the gray level
	for (int gray = 0; gray < 256; gray++)
	{
		int nSize = ImgCounter * hist._vBXY[gray].size();
		Mat blue(ImgCounter, hist._vBXY[gray].size(), CV_32FC1); blue.setTo(0);
		assert(ImgCounter == vImg.size());
		for (int i = 0; i < ImgCounter; i++)
		{
			for (int p = 0; p < hist._vBXY[gray].size(); p++)
			{
				short x = hist._vBXY[gray][p].x;
				short y = hist._vBXY[gray][p].y;
				blue.at<float>(i, p) = (vImg[i].ptr<uchar>(y) +3 * x)[0];
			}
		}
		vBlue.push_back(blue);
	}

	return;
}
void statistics(const Mat& meanInt, short ImgCounter, const vector<Mat>& vImg, const SHist& hist, vector<Mat>& vSoS)
{
	for (int gray = 0; gray < 256; gray++)
	{
		int nSize = ImgCounter * hist._vBXY[gray].size();
		Mat sos(ImgCounter, hist._vBXY[gray].size(), CV_32FC1); sos.setTo(0);
		assert(ImgCounter == vImg.size());
		for (int i = 0; i < ImgCounter; i++)
		{
			for (int p = 0; p < hist._vBXY[gray].size(); p++)
			{
				short x = hist._vBXY[gray][p].x;
				short y = hist._vBXY[gray][p].y;
				sos.at<float>(i, p) = (vImg[i].ptr<uchar>(y) +3 * x)[0] - meanInt.ptr<double>(0)[gray];
				sos.at<float>(i, p) *= sos.at<float>(i, p);
			}
		}
		vSoS.push_back(sos);
	}
	return;
}


void histo(const Mat& image_, SHist& hist_){
	for (int r = 0; r < image_.rows; r++)
	{
		for (int c = 0; c < image_.cols; c++)
		{
			const uchar* pBGR = image_.ptr<uchar>(r)+3 * c;
			hist_._vBXY[pBGR[0]].push_back(Point_<short>(c, r));
			hist_._vGXY[pBGR[1]].push_back(Point_<short>(c, r));
			hist_._vRXY[pBGR[2]].push_back(Point_<short>(c, r));
		}
	}
}

void total_statistics(
vector<Mat> avgImages, //average image per exposure
vector<vector<Mat>> vvImgs, //vvImgs[exp][n] exposure x number 
vector<SHist> vHist, // one histogram per exp
vector<float> times, // exposure time
vector<short> counter // number of image per exp
){
	vector<vector<Mat>> vvInten; //vvInten[exp][256]
	for (int e = 0; e < vvImgs.size(); e++)
	{
		vector<Mat> vInten;
		mean(counter[e], vvImgs[e], vHist[e], vInten);
		vvInten.push_back(vInten);
	}
	Mat meanInt(1, 256, CV_64FC1); meanInt.setTo(0);
	
	for (int g = 0; g < 256; g++){
		double m = 0;
		int s = 0;
		for (int e = 0; e < vvInten.size(); e++)
		{
			m += sum(vvInten[e][g])[0];
			s += vvInten[e][g].rows * vvInten[e][g].cols;
		}
		meanInt.ptr<double>(0)[g] = m/s;
	}
	cout << meanInt << endl;

	vector<Mat> vStd;
	for (int e = 0; e < vvImgs.size(); e++)
	{
		vector<Mat> vErr;
		statistics(meanInt, counter[e], vvImgs[e], vHist[e], vErr);

		Mat std(1, 256, CV_64FC1);
		double tE = 0; int nE = 0;
		for (int g = 0; g < 256; g++)
		{
			tE += sum(vErr[g])[0];
			nE += vErr[g].cols * vErr[g].rows;
			std.ptr<double>(0)[g] = sqrt(tE / nE);
		}
		vStd.push_back(std);
		cout << "std" <<e<< "=" << std << endl;
	}
}

void export(const Mat& invCrf){
	ofstream _out_crf;
	char outFileName[300];
	sprintf(outFileName, "..//load_inv_crf.m");
	_out_crf.open(outFileName);
	_out_crf << std::setprecision(15) << endl;
	_out_crf << "function [invCRF] = load_inv_crf()" << endl;
	_out_crf << "invCRF = [ ";
	for (int r = 0; r < invCrf.rows; r++)
	{
		_out_crf << invCrf.ptr<float>(r)[0] << ", " << invCrf.ptr<float>(r)[1] << ", " << invCrf.ptr<float>(r)[2] << ";" << endl;
	}
	_out_crf << "];";
	_out_crf.close();
}


void loadExposureSeq(vector<Mat>& images, vector<float>& times, vector<short>& ImgCounter, vector<vector<Mat>>& vvImg, vector<SHist>& vHist)
{
	float val = 1000.f;
	//vHist[expo]
	for (int expos = 3; expos < 193; expos <<= 1)
	{
		vector<Mat> vImg; //vImg[no];
		short nCout = 0;
		Mat sum(480, 640, CV_32FC3); sum.setTo(0.);
		//average out the sensor noise
		for (int i = 0; i < 30; i++)
		{
			ostringstream path;
			path << _path << expos << "." << i << ".png";
			Mat img = imread(path.str());
			if (img.empty()) continue;
			nCout++;
			vImg.push_back(img);
			Mat result;
			add(sum, img, result, Mat(), CV_32FC3);
			assert(result.type() == CV_32FC3);
			result.copyTo(sum);
		}
		vvImg.push_back(vImg);

		Mat imgAvg;
		sum.convertTo(imgAvg, CV_8UC3, 1.f / nCout, 0.5f);
		images.push_back(imgAvg);
		ostringstream path;
		path << _path << expos << ".png";
		imwrite(path.str(), imgAvg);
		times.push_back(expos / val);
		ImgCounter.push_back(nCout);
		cout << "expo " << expos << " # " << nCout << endl;

		//calc hist
		SHist hist;
		histo(imgAvg, hist);
		vHist.push_back(hist);
	}

	return;
}

string _path;
int main(int, char**argv)
{
	//update the location of the source images
	_path = string("C://csxsl//src//eccv16//hdr//hdr_capturer//1406120314//");

	vector<Mat> avgImages; //average image per exposure
	vector<vector<Mat>> vvImgs; //vvImgs[exp][n] exposure x number 
	vector<SHist> vHist; // one histogram per exp
	vector<float> times; // exposure time
	vector<short> counter; // number of image per exp
	//
	cout << "load images" << endl;
	loadExposureSeq(avgImages, times, counter, vvImgs, vHist);

	Mat response;
	Ptr<CalibrateDebevec> calibrate = createCalibrateDebevec(70,30,true);
	cout << "calc response function" << endl;
	calibrate->process(avgImages, response, times);
	export(response);
	Mat hdr;
	Ptr<MergeDebevec> merge_debevec = createMergeDebevec();
	merge_debevec->process(avgImages, hdr, times, response);
	imwrite("hdr.hdr", hdr);

	return 0; 
}

