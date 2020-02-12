#pragma once
#include <iostream>
#include <vector>
#include <cassert>
#include <math.h>
class FaceRecognition
{
private:
	std::vector<std::pair<std::vector<double> ,int>> gallerys;
	double dotProduct(const std::vector<double> v1, const std::vector<double> v2);
	double module(const std::vector<double> v);
	double cosine(const std::vector<double> v1, const std::vector<double> v2);
public:
	FaceRecognition();
	~FaceRecognition();
	int AddGallery(std::vector<double> feature,int label);
	std::pair<int, double> RecProbe(std::vector<double> feature);
	float CalcSimilarity(std::vector<double> f1,std::vector<double> f2);
	void ReleaseGallery();
};

