#include "FR.h"
#include "FaceAlignment.h"
#include "Funcation.h"
int main() {
	FR fr;
	Funcation fc;
	FaceAlignment fa;
	bool VIS = true;
	cv::VideoCapture cap;
	cap.open(0);
	cv::Mat frame;
	cap >> frame;
	
	int ex = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
	cv::VideoWriter video;
	video.open("demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), cap.get(cv::CAP_PROP_FPS), cv::Size(640, 480),true);
	
	//cv::Rect rect(frame.cols / 3, (frame.rows - frame.cols / 3) / 2, frame.cols / 3, frame.cols / 3);
	while (true && cv::waitKey(33)) {
		cap >> frame;
		if (frame.empty()) continue;
		//std::string result = fr.recProbe(fr.preCropImage(frame), "test");
		
		fr.detectFace(fr.preCropImage(frame));
		if (VIS) {
			cv::Mat detected_face = fr.getDetectedFace();
			if (!detected_face.empty()) {
				cv::Mat normalized_face = fr.getNomrlizedImage();
				cv::Rect detected_area = fr.getDetectedArea();
				detected_area.x /= 4; detected_area.y /= 4; detected_area.width /= 2; detected_area.height /= 2;
				detected_area.x += frame.cols / 3; detected_area.y += (frame.rows - frame.cols / 3) / 2;
				//cv::rectangle(frame,detected_area,cv::Scalar(0,255,0));

				cv::imshow("detected face", detected_face);
			}
		}
		cv::Mat result = fc.drawMask(frame);
		video << result;
		cv::imshow("frame", result);
	}
	//system("pause");
	cap.release();
	video.release();
}
