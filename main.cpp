#include "histogram.h"

int main()
{
	Mat srcImage, dstImage;
	srcImage = imread("1.png", 1);
	if (!srcImage.data) 
	{ 
		printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ��ͼƬ����~�� \n"); 
		return false; 
	}

	/* ��ά */
	// ת��Ϊ HLS��HIS�� ͨ��ͼ��
	Mat img;
	img = imread("1.jpg");
	Mat hlsimg;
	cvtColor(img, hlsimg, COLOR_RGB2HSV);
	drawHis(img, hlsimg);

	/* rgbֱ��ͼ */
	drawHisrgb(img);

	/* ֱ��ͼ�Ƚ� */
	Hisbijiao();

	/* ����ͶӰ */
	drawhr();

	/* ʵ��2�� */
	cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); // �ҶȻ�ͼ��
	drawHis1(srcImage, "ԭʼ�ĻҶ�ͼ��ֱ��ͼ"); // �滭һάֱ��ͼ
	imshow("ԭʼͼ", srcImage);
	equalizeHist(srcImage, dstImage);
	imshow("����ֱ��ͼ���⻯���ͼ", dstImage);
	drawHis1(dstImage, "���⻯��ͼ���ֱ��ͼ"); // �滭һάֱ��ͼ

	waitKey();
	destroyAllWindows();
	return 0;
}