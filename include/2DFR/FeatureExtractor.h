#pragma once
#include <iostream>
#include <fstream>
#include <map>
#include <future>
#include <string>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace mxnet::cpp;

class FeatureExtractor
{
private:
	Context ctx = Context::gpu();
	Context ctx_cpu = Context::gpu();
	map<string, NDArray> args_map;
	map<string, NDArray> aux_map;
	Symbol net;
	Executor *executor;
	void GetFeatureSymbol(string symbol_name, string layer_name);
	void GetFeatureSymbol(string symbol_name);
	void LoadParamtes(string model_name);
	NDArray Mat2NDArray(cv::Mat& image);
public:
	FeatureExtractor();
	int LoadModel(string net_name, string epoch, string layer_name,int image_size);
	std::vector<double> Extract(string image_name);
	std::vector<double> Extract(cv::Mat image);
	~FeatureExtractor();
};

