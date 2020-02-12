#pragma once
#include<opencv2/opencv.hpp>
#include<vector>
#include<string>

#include "FaceDetection.h"
#include "FeatureExtractor.h"
#include "FaceRecognition.h"
//����ͼ������ʾ���е���С���ռ��
#define MIN_FACE_AREA_RATIO 0.6
//1��N����ʶ�����������Ŷ���ֵ
#define REC_THRESHOLD 0.8
class FR
{
public:
	FR();
	~FR();
	int setGallery(std::vector<cv::Mat> images,std::vector<std::string> names,std::string task);
	std::string recProbe(cv::Mat image , std::string task);
	float verify(cv::Mat face_image, cv::Mat card_image);
	cv::Mat getNomrlizedImage();
	cv::Mat getDetectedFace();
	cv::Rect getDetectedArea();
	//Ԥ�ü�ͼ�񣬽�֧����recProbe֮ǰʹ��
	cv::Mat preCropImage(cv::Mat& frame);
	cv::Mat preCropCardImage(cv::Mat frame);
	//�ͷ�֮ǰ��ӵ�����
	void releaseGallery();
	void detectFace(cv::Mat& frame);
private:
	FaceDetection fd; 
	FaceRecognition fr;
	FeatureExtractor fe;
	cv::VideoCapture cap;
	cv::Mat normlized_image;
	cv::Mat detected_face;
	cv::Rect detected_area;
	std::vector<std::vector<double>> gallery_features;
	std::vector<std::string> gallery_names;
	cv::Mat detect(cv::Mat image);

};

