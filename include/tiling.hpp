#pragma once
#pragma once

#include "common.hpp"
#include <omp.h>
namespace cp
{
	//get online image size
	CP_EXPORT cv::Size getTileAlignSize(const cv::Size src, const cv::Size div_size, const int r, const int align_x, const int align_y, const int left_multiple = 1, const int top_multiple = 1);
	CP_EXPORT cv::Size getTileSize(const cv::Size src, const cv::Size div_size, const int r);

	//create a divided sub image
	CP_EXPORT void cropTile(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void cropTile(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void cropTileAlign(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int leftmultiple = 1, const int topmultiple = 1);

	CP_EXPORT void cropTile(const cv::Mat& src, cv::Mat& dest, const cv::Rect roi, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void cropTile(const cv::Mat& src, cv::Mat& dest, const cv::Rect roi, const int r, const int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void cropTileAlign(const cv::Mat& src, cv::Mat& dest, const cv::Rect roi, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int leftmultiple = 1, const int topmultiple = 1);

	CP_EXPORT void cropSplitTile(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const cv::Point idx, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType= cv::BORDER_DEFAULT);
	CP_EXPORT void cropSplitTileAlign(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int leftmultiple = 1, const int topmultiple = 1);

	//set a divided sub image to a large image
	CP_EXPORT void pasteTile(const cv::Mat& src, cv::Mat& dest,      const cv::Rect roi, const int top, const int left);
	CP_EXPORT void pasteTile(const cv::Mat& src, cv::Mat& dest,      const cv::Rect roi, const int r);
	CP_EXPORT void pasteTileAlign(const cv::Mat& src, cv::Mat& dest, const cv::Rect roi, const int r, const int left_multiple = 1, const int top_multiple = 1);

	CP_EXPORT void pasteTile(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int top, const int left);
	CP_EXPORT void pasteTile(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r);
	CP_EXPORT void pasteTileAlign(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int left_multiple = 1, const int top_multiple = 1);

	CP_EXPORT void pasteMergeTile(const std::vector <cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int top, const int left);
	CP_EXPORT void pasteMergeTile(const std::vector <cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r);
	CP_EXPORT void pasteMergeTileAlign(const std::vector <cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int left_multiple = 1, const int top_multiple = 1);

	//split an image to sub images in std::vector 
	CP_EXPORT void divideTiles(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const int r, const int borderType = cv::BORDER_DEFAULT);
	CP_EXPORT void divideTilesAlign(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int left_multiple = 1, const int top_multiple = 1);

	//merge subimages in std::vector to an image
	CP_EXPORT void conquerTiles(const std::vector<cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const int r);
	CP_EXPORT void conquerTilesAlign(const std::vector<cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const int r, const int left_multiple = 1, const int top_multiple = 1);

	class CP_EXPORT TileDivision
	{
		std::vector<cv::Point> pt;//left top point
		std::vector<cv::Size> tileSize;
		std::vector<int> threadnum;
		cv::Size div;
		cv::Size imgSize;
		int width_step = 0;
		int height_step = 0;

		void update_pt();
		bool isRecompute = true;
		bool preReturnFlag = false;
	public:

		//div.width * y + x;
		cv::Rect getROI(const int x, int y);
		cv::Rect getROI(const int index);

		TileDivision();
		void init(cv::Size imgSize, cv::Size div);
		TileDivision(cv::Size imgSize, cv::Size div);

		bool compute(const int width_step_, const int height_step_);

		void draw(cv::Mat& src, cv::Mat& dst);
		void draw(cv::Mat& src, cv::Mat& dst, std::vector<std::string>& info);
		void draw(cv::Mat& src, cv::Mat& dst, std::vector<std::string>& info, std::vector<std::string>& info2);
		void show(std::string wname);
	};

	class CP_EXPORT TileParallelBody
	{
		cp::TileDivision tdiv;
		cv::Size div;
		cv::Size tileSize;

		void init(const cv::Size div);
	protected:
		virtual void process(const cv::Mat& src, cv::Mat& dst, const int threadIndex, const int imageIndex) = 0;
		std::vector<cv::Mat> srcTile;
		std::vector<cv::Mat> dstTile;
		std::vector<cv::Mat> guideMaps;
		std::vector<std::vector<cv::Mat>> guideTile;
		int threadMax = omp_get_max_threads();
		bool isUseGuide = false;
		void initGuide(const cv::Size div, std::vector<cv::Mat>& guide);
	public:
		void drawMinMax(std::string wname, cv::Mat& src);

		void invoker(const cv::Size div, const cv::Mat& src, cv::Mat& dst, const int tileBoundary, const int borderType = cv::BORDER_DEFAULT, const int depth = -1);
		void unsetUseGuide();
		cv::Size getTileSize();
		//void printParameter();
	};
}