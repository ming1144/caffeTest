#include <iostream>
#include <fstream>
#include <iterator>
#include <string>

//MSVC
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#define _chdir(x) chdir(x)
#define _mkdir(x) mkdir(x,0777)
#endif

#define CPU_ONLY

#include <caffe\caffe.hpp>
#include "register.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <QtGui\QImage>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
public:
	Classifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file);

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);
	std::vector<Prediction> Classify(const QImage& img, int N = 5);

private:
	void SetMean(const string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);
	std::vector<float> Predict(const QImage& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img , std::vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;
	int num_channels_;
	std::vector<string> labels_; 

	int input_height_;
	int input_width_;
	QImage mean__;

	cv::Size input_geometry_;
	cv::Mat mean_;

};

Classifier::Classifier(const string& model_file,
	const string& trained_file,
	const string& mean_file,
	const string& label_file) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif
	//std::cout << model_file << std::endl << trained_file << std::endl;
	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	input_height_ = input_layer->height();
	input_width_  = input_layer->width();

	/* Load the binaryproto mean file. */
	SetMean(mean_file);

	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
	std::vector<float> output = Predict(img);

	N = std::min<int>(labels_.size(), N);
	std::vector<int> maxN = Argmax(output, N);
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
	}

	return predictions;
}

std::vector<Prediction> Classifier::Classify(const QImage& img, int N) {
	std::vector<float> output = Predict(img);

	N = std::min<int>(labels_.size(), N);
	std::vector<int> maxN = Argmax(output, N);
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
	}

	return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";
	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}
	

	float* data2 = mean_blob.mutable_cpu_data();
	float meandata[3] = {0,0,0};

	float* meanImage;
	meanImage = new float[num_channels_*mean_blob.height() * mean_blob.width()];
	for (int i = 0; i < num_channels_ * mean_blob.height() * mean_blob.width(); i++, data2++)
	{
		meanImage[i] = *data2;
		meandata[i / (mean_blob.height() * mean_blob.width())] += *data2;
	}

	for (int i = 0; i < num_channels_; i++)
	{
		meandata[i] /= (mean_blob.height()*mean_blob.width());
	}

	mean__ = QImage(mean_blob.width(), mean_blob.height(), QImage::Format_RGB32);
	if (num_channels_ == 3)
	{
		QRgb temp = qRgb(meandata[2], meandata[1], meandata[0]);
		mean__.fill(temp);
	}
	else
	{
		QRgb temp = qRgb(meandata[0], meandata[0], meandata[0]);
		mean__.fill(temp);
	}
	

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

std::vector<float> Classifier::Predict(const QImage& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	float* input_data = input_layer->mutable_cpu_data();
	float* data = (float*)img.bits();

	for (int y = 0; y < input_height_; y++)
	{
		for (int x = 0; x < input_width_; x++)
		{
			*input_data = *data;
			input_data++;
			data++;
		}
	}

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

int startImg = 1;
int endImg = 22;
int maskHeight = 47;
bool outsideTest = false;
bool useQT = true;

int main()
{
	int i, j, m;

	string modelType = std::to_string(maskHeight);
	modelType += "_SubMean";

	string trained_filename = "model_" + modelType + "/deploy.prototxt";
	string mean_filename = "model_" + modelType + "/mean.binaryproto";
	string model_filename = "model_" + modelType + "/model.caffemodel";
	string label_filename = "model_" + modelType + "/labels.txt";
	Classifier classifier(trained_filename, model_filename, mean_filename, label_filename);

	if (!outsideTest)
	{
		_mkdir(modelType.c_str());
		for (m = startImg; m <= endImg; m++)
		{
			string testImage_filename = "image/test";
			testImage_filename += std::to_string(m);
			testImage_filename += ".bmp";

			if (useQT)
			{
				Mat testImg = cv::imread(testImage_filename);
				Mat testImgGray(testImg);
				cv::cvtColor(testImg, testImgGray, CV_BGR2GRAY);

				Mat result(testImg.rows, testImg.cols, testImg.type(), Scalar(0));
				//Mat result = cv::imread(testImage_filename);

				for (i = maskHeight / 2; i < testImgGray.rows - maskHeight / 2; i++)
				{
					for (j = maskHeight / 2; j < testImgGray.cols - maskHeight / 2; j++)
					{
						cv::Point centerPoint;
						centerPoint.x = j;
						centerPoint.y = i;
						cv::Rect Region(Point(j - maskHeight / 2, i - maskHeight / 2), Point(j + maskHeight / 2 + 1, i + maskHeight / 2 + 1));
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
				string resultFilename = "result_";
				resultFilename += std::to_string(m);
				resultFilename += "_";
				resultFilename += modelType;
				resultFilename += ".PNG";
				imwrite(resultFilename, result);
				_chdir("..");
			}
			else
			{
				QImage testImg(QString::fromStdString(testImage_filename));
				QImage testImgGray(testImg);
				testImgGray.convertToFormat(QImage::Format_Grayscale8);

				QImage result(testImg.width(), testImg.height(), QImage::Format_Grayscale8);
				
				QRgb temp = qRgb(0, 0, 0);
				result.fill(temp);
				for (i = maskHeight / 2; i < testImgGray.height() - maskHeight / 2; i++)
				{
					for (j = maskHeight / 2; j < testImgGray.width() - maskHeight / 2; j++)
					{
						QRect rect(QPoint(j - maskHeight / 2, i - maskHeight / 2), QPoint(j + maskHeight / 2, i + maskHeight / 2));
						QImage ROI = testImgGray.copy(rect);

						std::vector<Prediction> predictions = classifier.Classify(ROI);

						Prediction temp = predictions[0];
						QRgb tempColor = qRgb(255, 255, 255);
						if (temp.first == "1")
						{
							result.setPixel(j, i, tempColor);
						}
					}
				}
				_chdir(modelType.c_str());
				string resultFilename = "result_";
				resultFilename += std::to_string(m);
				resultFilename += "_";
				resultFilename += modelType;
				resultFilename += ".PNG";
				result.save(QString::fromStdString(resultFilename));
				_chdir("..");

			}
		}
	}
	else
	{
		_chdir("outsideTestImage");
		string testImage_filename = "testImage.txt";
		std::vector<string> testImages;
		std::ifstream input(testImage_filename);

		char temp[100];

		while (input.getline(temp, sizeof(temp)))
		{
			string tempString(temp);
			testImages.push_back(tempString);
		}

		for (int m = 0; m < testImages.size(); m++)
		{
			Mat testImg = cv::imread(testImages[m]);
			Mat testImgGray(testImg);
			cvtColor(testImg, testImgGray, CV_BGR2GRAY);

			Mat result(testImg);

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

			string resultFilename = "result_";
			resultFilename += modelType;
			resultFilename += "_outsideTest-";
			resultFilename += std::to_string(m);
			resultFilename += ".PNG";
			imwrite(resultFilename, result);
		}
	}


	return 0;
}