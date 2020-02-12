#include "FaceDetection.h"
#include "FaceRecognition.h"
#include "FeatureExtractor.h"
#include <string>
void test_2dfr() {
	FaceDetection FD;
	FeatureExtractor FE;
	FaceRecognition FR;
	int face_size = 112;
	std::cout<<"init face detection..."<<std::endl;
	FD.LoadModel("models/rgb/face_seg", "0548", "segnet0_relu8_fwd_output");
	//std::cout<<"End"<<std::endl;
	//std::cout<<"init FeatureExtractor..."<<std::endl;
	//FE.LoadModel("models/rgb/mobileface", "0000", "conv_6dw7_7_conv2d_output", face_size);
	//std::cout<<"End"<<std::endl;
	cv::VideoCapture cap("data/tilted_face.avi");
	char key = '-1';
	int id_count = 0;
	std::vector<std::string> name_list;
	while (key != 's') {
		cv::Mat gray;
		cv::Mat frame ;
		//frame = cv::imread("data/img_135.jpg");
		cap >> frame;
		if (frame.empty()) break;
		std::cout<<frame.cols<<","<<frame.rows<<std::endl;
		cv::GaussianBlur(frame, frame, cv::Size(3, 3), 0);
		frame = FD.resize(frame, cv::Size(256, 256));
		cv::Mat image = frame.clone();
		std::vector<cv::Rect> rects = FD.Detect(frame.clone());
		FD.drawRect(image, rects);
		if (key == 'g') {
			std::string name = "null";
			//cv::imshow("face detected", frame(rects.at(0)));
			//cv::waitKey(33);
	
			std::cout << "Please input your name :\n";
			std::cin >> name;
			cv::imwrite("data/color_face/" + name + ".bmp", frame(rects.at(0)));
			
			//std::vector<double> feature = FE.Extract(frame(rects.at(0)));
			//FR.AddGallery(feature, id_count++);
			name_list.push_back(name);
		}

		//std::vector<double> feature = FE.Extract(frame(rects.at(0)));
		//std::pair<int, double> result = FR.RecProbe(feature);
		//std::cout << result.first << "\t" << result.second << std::endl;
		//if (result.first >= 0 && result.second >0.8) {
		//	cv::putText(image, name_list[result.first] + "," + std::to_string(int(result.second * 100)), cv::Point(rects[0].x, rects[0].y), 1, 1, cv::Scalar(255, 0, 255));
		//}
		cv::imshow("frame", image);
		key = cv::waitKey(33);
	}
}
int main() {
	test_2dfr();
}