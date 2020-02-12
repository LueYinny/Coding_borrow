#include "FaceDetectionDlib.h"
#include "FaceDetection.h"
#include "FeatureExtractor.h"
#include "FaceRecognition.h"
int main() {
	FaceDetectionDlib fdd;
	FaceDetection fd; fd.LoadModel("D:/code/2DFR_C++/models/rgb/face_seg", "0548", "segnet0_relu8_fwd_output",128);
	FeatureExtractor fe; fe.LoadModel("D:/code/2DFR_C++/models/rgb/mobileface", "0000", "conv_6dw7_7_conv2d_output", 112);
	FaceRecognition fr;
	cv::VideoCapture cap;
	cap.open(0);
	//cap.open("D:/code/2DFR_C++/data/tilted_face.avi");
	cv::Mat frame;
	int id_count = 0;
	std::vector<std::string> name_list;
	char key='0';

	while (true) {
		cap >> frame;
		if (frame.empty()) break;
		double t = (double)cv::getTickCount();
		cv::Mat frame_tmp = fd.resize(frame, cv::Size(512, 512));
		frame = fd.resize(frame, cv::Size(128, 128));
		std::vector<cv::Rect> rects = fd.Detect(frame);
		if (rects[0].width == 0 ||
			rects[0].height == 0 || 
			rects[0].x >=128 ||
			rects[0].y >=128 || 
			rects[0].width+ rects[0].y >128 ||
			rects[0].height + rects[0].x >128)
			continue;
		//for (int i = 0; i < rects.size(); i++) {
		//	if (rects[i].width == 0 || 
		//		rects[i].height == 0 || 
		//		rects[i].x >=128 ||
		//		rects[i].y >=128 || 
		//		(rects[i].width+ rects[i].y) >128 ||
		//		(rects[i].height + rects[i].x) >128) continue;
		//	cv::rectangle(frame,rects[i],cv::Scalar(255,0,0));
		//	//std::cout << rects[i].x << " " << rects[i].y  << " " << rects[i].width  << " " << rects[i].height  << std::endl;
		//	//std::cout << rects[i].x * 4 << " " << rects[i].y * 4 << " " << rects[i].width * 4 << " " << rects[i].height * 4 << std::endl;
		//	//cv::Mat face = frame_tmp(cv::Rect(rects[i].x*4,rects[i].y*4,rects[i].width*4,rects[i].height*4));
		//	//std::vector<cv::Point2d> lmk = fdd.localize(frame_tmp, cv::Rect(rects[i].x * 4, rects[i].y * 4, rects[i].width * 4, rects[i].height * 4));
		//	//for (int j = 0; j < lmk.size(); j++) {
		//	//	cv::circle(frame_tmp, lmk[j], 3, cv::Scalar(255, 0, 0));
		//	//}
		//}
		cv::Mat face = frame_tmp(cv::Rect(rects[0].x * 4, rects[0].y * 4, rects[0].width * 4, rects[0].height * 4));
		cv::resize(face, face, cv::Size(112, 112));

		if (key == 'g') {
			std::string name = "null";

			std::cout << "Please input your name :\n";
			std::cin >> name;
			cv::imwrite("D:/code/2DFR_C++/data/color_face/" + name + ".bmp", face);
			
			std::vector<double> feature = fe.Extract(face);
			fr.AddGallery(feature, name_list.size());
			name_list.push_back(name);
		}
		if (name_list.size() > 0) {
			std::vector<double> feature = fe.Extract(face);
			std::pair<int, double> result = fr.RecProbe(feature);
			//cv::rectangle(frame, rects[0], cv::Scalar(255, 255, 255));
			//std::cout << result.first << "\t" << result.second << std::endl;
			if (result.first >= 0 && result.second > 0.8) {
				cv::putText(frame_tmp, name_list[result.first] + "," + std::to_string(int(result.second * 100)), cv::Point(rects[0].x, rects[0].y), 1, 1, cv::Scalar(255, 0, 255));
			}
		}
		printf("time 2 = %f ms\n", 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency());
		cv::imshow("frame", frame_tmp);
		key = cv::waitKey(33);
	}
	return 0;
}