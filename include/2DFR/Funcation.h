#pragma once
#include<opencv2/opencv.hpp>
class Funcation
{
public:
	Funcation();
	~Funcation();
	cv::Mat drawMask(cv::Mat frame);
	cv::Mat drawCardMask(cv::Mat frame);

};

