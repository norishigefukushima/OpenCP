#pragma once

#include "common.hpp"
#include "plot.hpp"

namespace cp
{
	class CP_EXPORT SpearmanRankOrderCorrelationCoefficient
	{
		template<typename T>
		double mean(std::vector<T>& s);
		template<typename T>
		double covariance(std::vector<T>& s1, std::vector<T>& s2);
		template<typename T>
		double std_dev(std::vector<T>& v1);
		template<typename T>
		double pearson(std::vector<T>& v1, std::vector<T>& v2);
		template<typename T>
		void searchList(std::vector<T>& theArray, int sizeOfTheArray, double findFor, std::vector<int>& index);
		template<typename T>
		void Rank(std::vector<T>& vec, std::vector<T>& orig_vect, std::vector<T>& dest);
		template<typename T>
		double spearman_(std::vector<T>& v1, std::vector<T>& v2);

		void setPlotData(std::vector<float>& v1, std::vector<float>& v2, std::vector<cv::Point2d>& data);
		void setPlotData(std::vector<double>& v1, std::vector<double>& v2, std::vector<cv::Point2d>& data);

		cp::Plot pt;
		std::vector<cv::Point2d> plotsRAW;
		std::vector<cv::Point2d> plotsRANK;

	public:

		double spearman(std::vector<float> v1, std::vector<float> v2);//compute SROCC (vector<float>). 
		double spearman(std::vector<double> v1, std::vector<double> v2);//compute SROCC ((vector<double>)). 
		void plot(const bool isWait = true);
	};
}
