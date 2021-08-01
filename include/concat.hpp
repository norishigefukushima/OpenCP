#pragma once

#include "common.hpp"

namespace cp
{
	//extention of cv::hconcat and cv::vconcat. (tile_cols>=1, tile_rows >= 1)
	CP_EXPORT void concat(cv::InputArrayOfArrays src, cv::OutputArray dest, const int tile_cols, const int tile_rows = 1);
	//same function of concat for consistency of notation: tile_cols>=1, tile_rows >= 1
	CP_EXPORT void concatMerge(cv::InputArrayOfArrays src, cv::OutputArray dest, const int tile_cols, const int tile_rows = 1);
	
	//inverse function of concat (concatted image to vector<Mat>).
	CP_EXPORT void concatSplit(cv::InputArray src, cv::OutputArrayOfArrays dest, const cv::Size image_size);
	//inverse function of concat (concatted image to vector<Mat>).
	CP_EXPORT void concatSplit(cv::InputArray src, cv::OutputArrayOfArrays dest, const int tile_cols, const int tile_rows);

	//tile_col_index>=0, tile_row_index >= 0
	CP_EXPORT void concatExtract(cv::InputArrayOfArrays src, cv::OutputArray dest, const int tile_col_index, const int tile_row_index = 0);
	//if dest is empty: tile_col_index>=0, tile_row_index >= 0
	CP_EXPORT void concatExtract(cv::InputArrayOfArrays src, cv::Size size, cv::OutputArray dest, const int tile_col_index, const int tile_row_index = 0);
}