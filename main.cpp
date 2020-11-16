#include "histogram.h"

int main()
{
	Mat srcImage, dstImage;
	srcImage = imread("1.png", 1);
	if (!srcImage.data) 
	{ 
		printf("读取图片错误，请确定目录下是否有imread函数指定图片存在~！ \n"); 
		return false; 
	}

	/* 二维 */
	// 转换为 HLS（HIS） 通道图像
	Mat img;
	img = imread("1.jpg");
	Mat hlsimg;
	cvtColor(img, hlsimg, COLOR_RGB2HSV);
	drawHis(img, hlsimg);

	/* rgb直方图 */
	drawHisrgb(img);

	/* 直方图比较 */
	Hisbijiao();

	/* 反向投影 */
	drawhr();

	/* 实验2补 */
	cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); // 灰度化图像
	drawHis1(srcImage, "原始的灰度图的直方图"); // 绘画一维直方图
	imshow("原始图", srcImage);
	equalizeHist(srcImage, dstImage);
	imshow("经过直方图均衡化后的图", dstImage);
	drawHis1(dstImage, "均衡化后图像的直方图"); // 绘画一维直方图

	waitKey();
	destroyAllWindows();
	return 0;
}