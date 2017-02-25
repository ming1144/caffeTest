#include "classifier.h"
#include "register.h"

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <direct.h>

#define CPU_ONLY

#include <caffe\caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <QtGui\QImage>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;

int startImg = 1;
int endImg = 22;
int maskHeight = 33;
bool outsideTest = true;
bool padding = false;
bool withOriginImage = false;

int main()
{
	int i, j, m;

	//string modelType = std::to_string(maskHeight);
	//modelType += "_SubMean";
	
	string modelType = "CT";

	if (padding)
	{
		modelType += "_padding";
	}

	string trained_filename = "model_" + modelType + "/deploy.prototxt";
	string mean_filename = "model_" + modelType + "/mean.binaryproto";
	string model_filename = "model_" + modelType + "/model.caffemodel";
	string label_filename = "model_" + modelType + "/labels.txt";
	Classifier classifier(trained_filename, model_filename, mean_filename, label_filename);

	if (!outsideTest)
	{
		_chdir("trainingData");
		_chdir("female");
		_mkdir("result");
		string testImage_filename = "testfile.txt";
		std::vector<string> testImages;
		std::ifstream input(testImage_filename);

		string temp;

		while (getline(input, temp))
		{
			string temp2;
			getline(input, temp2);
			testImages.push_back(temp2);
		}
		for ( m = 0; m < testImages.size(); m++)
		{
			Mat *testImg = &cv::imread(testImages[m].c_str());
			if (testImg->empty())
			{
				std::cout << "open Image error:" << m << std::endl;
				continue;
			}
			Mat testImgGray(*testImg);
			cvtColor(*testImg, testImgGray, CV_BGR2GRAY);

			
			Mat result;

			if (!withOriginImage)
			{
				result = Mat(testImg->rows, testImg->cols, testImg->type(), Scalar(0));
			}
			else
			{
				result = Mat(testImgGray);
			}
			
			for (i = maskHeight / 2; i < testImgGray.rows - maskHeight / 2; i++)
			{
				for (j = maskHeight / 2; j < testImgGray.cols - maskHeight / 2; j++)
				{
					Point centerPoint;
					centerPoint.x = j;
					centerPoint.y = i;
					Rect Region(Point(j - maskHeight / 2, i - maskHeight / 2), Point(j + maskHeight / 2 + 1, i + maskHeight / 2 + 1));
					Mat ROI(testImgGray, Region);

					std::vector<Prediction> predictions = classifier.Classify(ROI);

					Prediction temp = predictions[0];
					if (temp.first == "1")
					{
						result.at<Vec3b>(centerPoint) = Vec3b::all(255);
					}
				}
			}
			_chdir(modelType.c_str());
			string resultFilename = "result/";
			resultFilename += std::to_string(m);
			resultFilename += "_";
			resultFilename += modelType;
			resultFilename += ".PNG";
			cv::imwrite(resultFilename, result);
			_chdir("..");
			testImg->deallocate();
		}
	}

	else
	{
		/*_chdir("trainingData");
		_chdir("female");*/

		_chdir("CT_test");
		_mkdir("result");
		string testImage_filename = "testfile.txt";

		std::vector<string> testImages;
		std::ifstream input(testImage_filename);

		string temp;

		/*while (getline(input,temp))
		{
			string temp2;
			getline(input, temp2);
			testImages.push_back(temp2);
		}*/

		while (getline(input, temp))
		{
			testImages.push_back(temp);
		}

		for (int m = 0; m < testImages.size(); m++)
		{
			Mat testImg = cv::imread(testImages[m]);
			Mat testImgGray(testImg);
			cvtColor(testImg, testImgGray, CV_BGR2GRAY);

			Mat result;
			if (!withOriginImage)
			{
				result = Mat(testImg.rows, testImg.cols, testImg.type(), Scalar(0));
			}
			else
			{
				result = Mat(testImg);
			}
			for (i = maskHeight / 2; i < testImgGray.rows - maskHeight / 2; i++)
			{
				for (j = maskHeight / 2; j < testImgGray.cols - maskHeight / 2; j++)
				{
					Point centerPoint;
					centerPoint.x = j;
					centerPoint.y = i;
					Rect Region(Point(j - maskHeight / 2, i - maskHeight / 2), Point(j + maskHeight / 2 + 1, i + maskHeight / 2 + 1));
					Mat ROI(testImgGray, Region);

					std::vector<Prediction> predictions = classifier.Classify(ROI);

					Prediction temp = predictions[0];
					if (temp.first == "1")
					{
						result.at<Vec3b>(centerPoint) = Vec3b::all(255);
					}
				}
			}

			//string resultFilename = "result_";
			//resultFilename += modelType;
			//resultFilename += "_outsideTest-";
			//resultFilename += std::to_string(m);
			//resultFilename += ".PNG";
			//cv::imwrite(resultFilename, result);

			string resultFilename = "result/";
			resultFilename += testImages[m];
			cv::imwrite(resultFilename, result);
		}
	}

	return 0;
}