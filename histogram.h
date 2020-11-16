#pragma once
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

#define WINDOW_NAME1 "��ԭʼͼ��" 

void drawHis(Mat srcImage, Mat dstImage)
{
	//��ɫ������Ϊ30���ȼ��������Ͷ�����Ϊ32���ȼ�
	int hueBinNum = 30;//ɫ����ֱ��ͼֱ������
	int saturationBinNum = 32;//���Ͷȵ�ֱ��ͼֱ������
	int histSize[] = { hueBinNum, saturationBinNum };
	// ����ɫ���ı仯��ΧΪ0��179
	float hueRanges[] = { 0, 180 };
	//���履�Ͷȵı仯��ΧΪ0���ڡ��ס��ң���255����������ɫ��
	float saturationRanges[] = { 0, 256 };
	const float* ranges[] = { hueRanges, saturationRanges };
	MatND dstHist;
	//����׼����calcHist�����н������0ͨ���͵�1ͨ����ֱ��ͼ
	int channels[] = { 0, 1 };

	//��3����ʽ����calcHist������ֱ��ͼ����
	calcHist(&dstImage,//���������
		1, //�������Ϊ1
		channels,//ͨ������
		Mat(), //��ʹ����Ĥ
		dstHist, //�����Ŀ��ֱ��ͼ
		2, //��Ҫ�����ֱ��ͼ��ά��Ϊ2
		histSize, //���ÿ��ά�ȵ�ֱ��ͼ�ߴ������
		ranges,//ÿһά��ֵ��ȡֵ��Χ����
		true, // ָʾֱ��ͼ�Ƿ���ȵı�ʶ����true��ʾ���ȵ�ֱ��ͼ
		false);//�ۼƱ�ʶ����false��ʾֱ��ͼ�����ý׶λᱻ����

	//��4��Ϊ����ֱ��ͼ׼������
	double maxValue = 0;//���ֵ
	minMaxLoc(dstHist, 0, &maxValue, 0, 0);//����������������ȫ����Сֵ�����ֵ����maxValue��
	int scale = 10;
	Mat histImg = Mat::zeros(saturationBinNum * scale, hueBinNum * 10, CV_8UC3);

	//��5��˫��ѭ��������ֱ��ͼ����
	for (int hue = 0; hue < hueBinNum; hue++)
		for (int saturation = 0; saturation < saturationBinNum; saturation++)
		{
			float binValue = dstHist.at<float>(hue, saturation);//ֱ��ͼ����ֵ
			int intensity = cvRound(binValue * 255 / maxValue);//ǿ��

			//��ʽ���л���
			rectangle(histImg, Point(hue * scale, saturation * scale),
				Point((hue + 1) * scale - 1, (saturation + 1) * scale - 1),
				Scalar::all(intensity), FILLED);
		}

	//��6����ʾЧ��ͼ
	imshow("�ز�ͼ", srcImage);
	imshow("H-S ֱ��ͼ", histImg);
}

void drawHis1(Mat srcImage, string name)
{
	MatND dstHist;       // ��cv����CvHistogram *hist = cvCreateHist
	int dims = 1;		 // һά
	float hranges[] = { 0, 255 };	// �Ҷ�����
	const float* ranges[] = { hranges };   // ������ҪΪconst����
	int size = 256;  // �Ҷȼ���
	int channels = 0; 

	calcHist(&srcImage, 1, &channels, Mat(), dstHist, dims, &size, ranges);    // cv ����cvCalcHist
	int scale = 1;

	Mat dstImage(size * scale, size, CV_8U, Scalar(0)); // ȫ0����

	double minValue = 0;
	double maxValue = 0;
	minMaxLoc(dstHist, &minValue, &maxValue, 0, 0);  //  ��cv���õ���cvGetMinMaxHistValue

	int hpt = saturate_cast<int>(0.9 * size);
	for (int i = 0; i < 256; i++)
	{
		float binValue = dstHist.at<float>(i);           //   ע��hist����float����    ����OpenCV1.0������cvQueryHistValue_1D
		int realValue = saturate_cast<int>(binValue * hpt / maxValue);
		rectangle(dstImage, Point(i * scale, size - 1), Point((i + 1) * scale - 1, size - realValue), Scalar(255));
	}
	imshow(name, dstImage);
}

void drawHisrgb(Mat srcImage)
{
	int bins = 256;
	int hist_size[] = { bins };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	MatND redHist, grayHist, blueHist;
	int channels_r[] = { 0 };

	//��3������ֱ��ͼ�ļ��㣨��ɫ�������֣�
	calcHist(&srcImage, 1, channels_r, Mat(), //��ʹ����Ĥ
		redHist, 1, hist_size, ranges,
		true, false);

	//��4������ֱ��ͼ�ļ��㣨��ɫ�������֣�
	int channels_g[] = { 1 };
	calcHist(&srcImage, 1, channels_g, Mat(), // do not use mask
		grayHist, 1, hist_size, ranges,
		true, // the histogram is uniform
		false);

	//��5������ֱ��ͼ�ļ��㣨��ɫ�������֣�
	int channels_b[] = { 2 };
	calcHist(&srcImage, 1, channels_b, Mat(), // do not use mask
		blueHist, 1, hist_size, ranges,
		true, // the histogram is uniform
		false);

	//-----------------------���Ƴ���ɫֱ��ͼ------------------------
	//����׼��
	double maxValue_red, maxValue_green, maxValue_blue;
	minMaxLoc(redHist, 0, &maxValue_red, 0, 0);
	minMaxLoc(grayHist, 0, &maxValue_green, 0, 0);
	minMaxLoc(blueHist, 0, &maxValue_blue, 0, 0);
	int scale = 1;
	int histHeight = 256;
	Mat histImage = Mat::zeros(histHeight, bins * 3, CV_8UC3);

	//��ʽ��ʼ����
	for (int i = 0; i < bins; i++)
	{
		//����׼��
		float binValue_red = redHist.at<float>(i);
		float binValue_green = grayHist.at<float>(i);
		float binValue_blue = blueHist.at<float>(i);
		int intensity_red = cvRound(binValue_red * histHeight / maxValue_red);  //Ҫ���Ƶĸ߶�
		int intensity_green = cvRound(binValue_green * histHeight / maxValue_green);  //Ҫ���Ƶĸ߶�
		int intensity_blue = cvRound(binValue_blue * histHeight / maxValue_blue);  //Ҫ���Ƶĸ߶�

		//���ƺ�ɫ������ֱ��ͼ
		rectangle(histImage, Point(i * scale, histHeight - 1),
			Point((i + 1) * scale - 1, histHeight - intensity_red),
			Scalar(255, 0, 0));

		//������ɫ������ֱ��ͼ
		rectangle(histImage, Point((i + bins) * scale, histHeight - 1),
			Point((i + bins + 1) * scale - 1, histHeight - intensity_green),
			Scalar(0, 255, 0));

		//������ɫ������ֱ��ͼ
		rectangle(histImage, Point((i + bins * 2) * scale, histHeight - 1),
			Point((i + bins * 2 + 1) * scale - 1, histHeight - intensity_blue),
			Scalar(0, 0, 255));

	}

	//�ڴ�������ʾ�����ƺõ�ֱ��ͼ
	imshow("ͼ���RGBֱ��ͼ", histImage);
}

void Hisbijiao()
{
	Mat srcImage_base, hsvImage_base;
	Mat srcImage_test1, hsvImage_test1;
	Mat srcImage_test2, hsvImage_test2;
	Mat hsvImage_halfDown;

	//��2�������׼ͼ��(srcImage_base) �����Ų���ͼ��srcImage_test1��srcImage_test2������ʾ
	srcImage_base = imread("1.jpg", 1);
	srcImage_test1 = imread("2.jpg", 1);
	srcImage_test2 = imread("3.jpg", 1);
	//��ʾ�����3��ͼ��
	imshow("��׼ͼ��", srcImage_base);
	imshow("����ͼ��1", srcImage_test1);
	imshow("����ͼ��2", srcImage_test2);

	// ��3����ͼ����BGRɫ�ʿռ�ת���� HSVɫ�ʿռ�
	cvtColor(srcImage_base, hsvImage_base, COLOR_BGR2HSV);
	cvtColor(srcImage_test1, hsvImage_test1, COLOR_BGR2HSV);
	cvtColor(srcImage_test2, hsvImage_test2, COLOR_BGR2HSV);

	//��4������������׼ͼ���°벿�İ���ͼ��(HSV��ʽ)
	hsvImage_halfDown = hsvImage_base(Range(hsvImage_base.rows / 2, hsvImage_base.rows - 1), Range(0, hsvImage_base.cols - 1));

	//��5����ʼ������ֱ��ͼ��Ҫ��ʵ��
	// ��hueͨ��ʹ��30��bin,��saturatoinͨ��ʹ��32��bin
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };
	// hue��ȡֵ��Χ��0��256, saturationȡֵ��Χ��0��180
	float h_ranges[] = { 0, 256 };
	float s_ranges[] = { 0, 180 };
	const float* ranges[] = { h_ranges, s_ranges };
	// ʹ�õ�0�͵�1ͨ��
	int channels[] = { 0, 1 };

	// ��6����������ֱ��ͼ�� MatND ���ʵ��:
	MatND baseHist;
	MatND halfDownHist;
	MatND testHist1;
	MatND testHist2;

	// ��7�������׼ͼ�����Ų���ͼ�񣬰����׼ͼ���HSVֱ��ͼ:
	calcHist(&hsvImage_base, 1, channels, Mat(), baseHist, 2, histSize, ranges, true, false);
	normalize(baseHist, baseHist, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsvImage_halfDown, 1, channels, Mat(), halfDownHist, 2, histSize, ranges, true, false);
	normalize(halfDownHist, halfDownHist, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsvImage_test1, 1, channels, Mat(), testHist1, 2, histSize, ranges, true, false);
	normalize(testHist1, testHist1, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsvImage_test2, 1, channels, Mat(), testHist2, 2, histSize, ranges, true, false);
	normalize(testHist2, testHist2, 0, 1, NORM_MINMAX, -1, Mat());


	//��8����˳��ʹ��4�ֶԱȱ�׼����׼ͼ���ֱ��ͼ�������ֱ��ͼ���жԱ�:
	for (int i = 0; i < 4; i++)
	{
		//����ͼ��ֱ��ͼ�ĶԱ�
		int compare_method = i;
		double base_base = compareHist(baseHist, baseHist, compare_method);
		double base_half = compareHist(baseHist, halfDownHist, compare_method);
		double base_test1 = compareHist(baseHist, testHist1, compare_method);
		double base_test2 = compareHist(baseHist, testHist2, compare_method);
		//������
		printf(" ���� [%d] ��ƥ�������£�\n\n ����׼ͼ - ��׼ͼ����%f, ����׼ͼ - ����ͼ����%f,����׼ͼ - ����ͼ1���� %f, ����׼ͼ - ����ͼ2����%f \n-----------------------------------------------------------------\n", i, base_base, base_half, base_test1, base_test2);
	}

	printf("��������");
}

Mat g_srcImage; Mat g_hsvImage; Mat g_hueImage;
int g_bins = 30;//ֱ��ͼ���
void on_BinChange(int, void*)
{
	//��1������׼��
	MatND hist;
	int histSize = MAX(g_bins, 2);
	float hue_range[] = { 0, 180 };
	const float* ranges = { hue_range };

	//��2������ֱ��ͼ����һ��
	calcHist(&g_hueImage, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

	//��3�����㷴��ͶӰ
	MatND backproj;
	calcBackProject(&g_hueImage, 1, 0, hist, backproj, &ranges, 1, true);

	//��4����ʾ����ͶӰ
	imshow("����ͶӰͼ", backproj);

	//��5������ֱ��ͼ�Ĳ���׼��
	int w = 400; int h = 400;
	int bin_w = cvRound((double)w / histSize);
	Mat histImg = Mat::zeros(w, h, CV_8UC3);

	//��6������ֱ��ͼ
	for (int i = 0; i < g_bins; i++)
	{
		rectangle(histImg, Point(i * bin_w, h), Point((i + 1) * bin_w, h - cvRound(hist.at<float>(i) * h / 255.0)), Scalar(100, 123, 255), -1);
	}

	//��7����ʾֱ��ͼ����
	imshow("ֱ��ͼ", histImg);
}

void drawhr()
{
	//��1����ȡԴͼ�񣬲�ת���� HSV �ռ�
	g_srcImage = imread("3.png", 1);
	if (!g_srcImage.data) 
	{ 
		printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ��ͼƬ����~�� \n"); 
		// return false; 
	}
	cvtColor(g_srcImage, g_hsvImage, COLOR_BGR2HSV);

	//��2������ Hue ɫ��ͨ��
	g_hueImage.create(g_hsvImage.size(), g_hsvImage.depth());
	int ch[] = { 0, 0 };
	mixChannels(&g_hsvImage, 1, &g_hueImage, 1, ch, 1);

	//��3������ Trackbar ������bin����Ŀ
	namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
	createTrackbar("ɫ����� ", WINDOW_NAME1, &g_bins, 180, on_BinChange);
	on_BinChange(0, 0);//����һ�γ�ʼ��

	//��4����ʾЧ��ͼ
	imshow(WINDOW_NAME1, g_srcImage);

}