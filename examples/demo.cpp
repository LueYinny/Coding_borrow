#include "FR.h"
#include "FaceAlignment.h"
#include "Funcation.h"
int main() {
	FR fr;
	Funcation fc;
	FaceAlignment fa;
	bool VIS = true;
	std::vector<std::string> gallery_list = {
		"data/gallery/Aaron_Eckhart_0001.jpg",
		"data/gallery/Aaron_Guiel_0001.jpg",
		"data/gallery/Aaron_Patterson_0001.jpg",
		"data/gallery/Aaron_Peirsol_0001.jpg",
		"data/gallery/Aaron_Pena_0001.jpg",
		"data/gallery/Aaron_Sorkin_0002.jpg",
		"data/gallery/Aaron_Tippin_0001.jpg",
		"data/gallery/Abba_Eban_0001.jpg",
		"data/gallery/muguodng.jpg"
	};
	std::vector<std::string> name_list = {
		"Aaron_Eckhart",
		"Aaron_Guiel",
		"Aaron_Patterson",
		"Aaron_Peirsol",
		"Aaron_Pena",
		"Aaron_Sorkin",
		"Aaron_Tippin",
		"Abba_Eban",
		"muguodong"
	};
	std::vector<std::string> probe_list = {
		"data/probe/Aaron_Peirsol_0002.jpg",
		"data/probe/Aaron_Peirsol_0003.jpg",
		"data/probe/Aaron_Peirsol_0004.jpg",
		"data/probe/Aaron_Sorkin_0001.jpg"
	};

	std::vector<cv::Mat> gallery_faces, probe_faces;
	for (int i = 0; i < gallery_list.size(); i++) {
		cv::Mat image = cv::imread(gallery_list[i]);
		gallery_faces.push_back(image);
	}
	fr.setGallery(gallery_faces,name_list,"test");
	//fr.releaseGallery();
	for (int i = 0; i < probe_list.size(); i++) {
		cv::Mat image = cv::imread(probe_list[i]);
		double t = (double)cv::getTickCount();
		std::string result = fr.recProbe(image,"test");
		std::cout << result << std::endl;
		printf("time cost = %f ms\n", 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency());

		//
		/*if (VIS) {
			cv::Mat detected_face = fr.getDetectedFace();
			cv::Mat normalized_face = fr.getNomrlizedImage();
			cv::Rect detected_area = fr.getDetectedArea();
			if (!detected_face.empty()) {
				std::vector<cv::Point2d> lmks = fdd.localize(normalized_face, detected_area);
				for (int l = 0; l < lmks.size(); l++) {
					cv::circle(normalized_face,lmks[l],2,cv::Scalar(0,255,255));
				}
				cv::imshow("normalized_face", normalized_face);
				cv::waitKey(0);
			}
		}*/
	}
	for (int i = 0; i < probe_list.size(); i++) {
		cv::Mat gallery = cv::imread(probe_list[i]);
		for (int j = i + 1; j < probe_list.size(); j++) {
			cv::Mat probe = cv::imread(probe_list[j]);
			float sim = fr.verify(probe,gallery);
			std::cout << "sim = "<< sim << std::endl;
			//
			if (VIS) {
				cv::Mat detected_face = fr.getDetectedFace();
				if (!detected_face.empty()) {
					cv::imshow("detected face", fr.getDetectedFace());
					cv::waitKey(500);
				}
			}
		}
	}
	cv::VideoCapture cap;
	//cap.open("data/face.avi");
	cap.open(0);
	cv::Mat frame;
	cap >> frame;
	cv::Rect rect( frame.cols / 3, (frame.rows - frame.cols / 3) / 2, frame.cols / 3, frame.cols / 3);
	while (true) {
		cap >> frame;
		if (frame.empty()) continue;
		cv::rectangle(frame, rect, cv::Scalar(0, 0, 255));
		cv::imshow("frame",fc.drawMask( frame));
		cv::waitKey(33);
		//std::cout << frame.cols << " " << frame.rows << std::endl;
		
		std::string result = fr.recProbe(fr.preCropImage(frame),"test");
		std::cout << "rec id  = "<<result << std::endl;
		if (VIS) {
			cv::Mat detected_face = fr.getDetectedFace();
			if (!detected_face.empty()) {
				if (VIS) {
					cv::Mat detected_face = fr.getDetectedFace();
					cv::Mat normalized_face = fr.getNomrlizedImage();
					cv::Rect detected_area = fr.getDetectedArea();
					if (!detected_face.empty()) {
						std::vector<cv::Point2d> lmks = fa.localize(normalized_face, detected_area);
						for (int l = 0; l < lmks.size(); l++) {
							cv::circle(normalized_face, lmks[l], 2, cv::Scalar(0, 255, 255));
						}
						cv::imshow("normalized_face", normalized_face);
						cv::imshow("detected face", fr.getDetectedFace());
						cv::waitKey(33);
					}
				}
			}
		}
	}
	//system("pause");
}
