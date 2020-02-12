#include "FaceDetection.h"
#include "FaceRecognition.h"
#include "FeatureExtractor.h"
#include <string>
void test_2dfr() {
	FaceDetection FD;
	FeatureExtractor FE;
	FaceRecognition FR;
	int face_size = 112;
	std::cout << "init face detection..." << std::endl;

	FD.LoadModel("D:/code/2DFR_C++/models/rgb/face_seg", "0548", "segnet0_relu8_fwd_output");
	FE.LoadModel("D:/code/2DFR_C++/models/rgb/mobileface", "0000", "conv_6dw7_7_conv2d_output", face_size);
	//FE.LoadModel("D:/code/2DFR_C++/models/rgb/msff_net", "0000", "s5_global_conv_output",128);

	cv::VideoCapture cap(0);
	char key = '-1';
	int id_count = 0;
	std::vector<std::string> name_list;
	while (key != 's') {
		cv::Mat frame;
		cv::Mat gray;
		cap >> frame;
		if (frame.empty()) break;
		cv::GaussianBlur(frame,frame,cv::Size(3,3),0);
		//cv::Mat & image = frame.clone();
		std::vector<cv::Rect> rects = FD.Detect(frame);
		if (rects.empty()) continue;
		//cv::imshow("face", face);
		cv::Mat color_face = frame(rects[0]);
		cv::resize(color_face, color_face, cv::Size(128, 128));
		if (key == 'g') {
			std::string name = "null";
			std::cout << "Please input your name :\n";
			std::cin >> name;
			cv::imwrite("data/color_face/" + name + ".bmp", frame(rects[0]));

			std::vector<double> feature = FE.Extract(frame(rects[0]));
			FR.AddGallery(feature, id_count++);
			name_list.push_back(name);
		}

		std::vector<double> feature = FE.Extract(color_face);
		std::pair<int, double> result = FR.RecProbe(feature);
		cv::rectangle(frame,rects[0],cv::Scalar(255,255,255));
		if (result.first >= 0 && result.second >0.85) {
			cv::putText(frame, name_list[result.first] + "," + std::to_string(int(result.second * 100)), cv::Point(rects[0].x, rects[0].y), 1, 2, cv::Scalar(0, 0, 255));
		}
		cv::imshow("frame", frame);
		key = cv::waitKey(33);
		/*
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		std::pair<cv::Rect, std::vector<cv::Point2f>> detected_face = FL.detectFaceAndNTP(gray, true);
		cv::Rect bbx;
		if (detected_face.first.width == 0 || detected_face.first.height == 0) {
			continue;
		}
		else {
			bbx = detected_face.first;
			if (bbx.width < bbx.height) bbx.width = bbx.height;
			else bbx.height = bbx.width;

			cv::rectangle(frame, bbx, -1);
			for (int l = 0; l < detected_face.second.size(); l++) {
				cv::circle(frame, detected_face.second.at(l), 3, -1);
			}

		}
		

		cv::Mat face = image(bbx);
		cv::resize(face, face, cv::Size(face_size, face_size));

		if (key == 'g') {
			std::string name = "null";
			std::cout << "Please input your name :\n";
			std::cin >> name;
			cv::imwrite("data/color_face/" + name + ".bmp", face);

			std::vector<double> feature = FE.Extract(face);
			FR.AddGallery(feature, id_count++);
			name_list.push_back(name);
		}

		std::vector<double> feature = FE.Extract(face);
		std::pair<int, double> result = FR.RecProbe(feature);
		if (result.first >= 0) {
			cv::putText(frame, name_list[result.first] + "," + std::to_string(int(result.second*100)), cv::Point(detected_face.first.x, detected_face.first.y), 1, 2, cv::Scalar(0, 0,255));
		}

		std::vector<cv::Point2f> face_lmk;
		face_lmk.push_back(cv::Point2f((detected_face.second[0].x- detected_face.first.x)/ float(detected_face.first.width)*face_size, (detected_face.second[0].y - detected_face.first.y)/ float(detected_face.first.height)*face_size));
		face_lmk.push_back(cv::Point2f((detected_face.second[1].x - detected_face.first.x) / float(detected_face.first.width)*face_size, (detected_face.second[1].y - detected_face.first.y) / float(detected_face.first.height)*face_size));
		face_lmk.push_back(cv::Point2f((detected_face.second[2].x - detected_face.first.x) / float(detected_face.first.width)*face_size, (detected_face.second[2].y - detected_face.first.y) / float(detected_face.first.height)*face_size));
		face_lmk.push_back(cv::Point2f((detected_face.second[3].x - detected_face.first.x) / float(detected_face.first.width)*face_size, (detected_face.second[3].y - detected_face.first.y) / float(detected_face.first.height)*face_size));
		face_lmk.push_back(cv::Point2f((detected_face.second[4].x - detected_face.first.x) / float(detected_face.first.width)*face_size, (detected_face.second[4].y - detected_face.first.y) / float(detected_face.first.height)*face_size));

		for (int l = 0; l < landmark_template.size(); l++) {
			cv::circle(face, landmark_template.at(l), 1, cv::Scalar(0,255, 0));
			cv::circle(face,face_lmk.at(l), 5, cv::Scalar(0, 0, 255));
			std::cout << cv::Point(face_lmk.at(l).x, face_lmk.at(l).y) << std::endl;
		}
		//cv::Mat warp_mat = cv::estimateRigidTransform(face_lmk, landmark_template, false);//使用相似变换，不适合使用仿射变换，会导致图像变形
		//cv::Mat warp_mat=cv::getAffineTransform(face_lmk, landmark_template);
		cv::Mat warp_mat = cv::estimateAffine2D(detected_face.second, landmark_template);
		cv::Mat warp_frame;
		warpAffine(image, warp_frame, warp_mat, cv::Size(face_size,face_size));//裁剪图像
		//cv::imshow("face", face);
		cv::imshow("warp_frame", warp_frame);
		cv::imshow("frame", frame);
		key = cv::waitKey(33);
		*/
	}
}
//int main() {
//	test_2dfr();
//}