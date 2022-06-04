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
		double spearman_(std::vector<T>& v1, std::vector<T>& v2, const bool ignoreTie, const bool isPlot);

		template<typename T>
		void rankTransformIgnoreTie(std::vector<T>& src, std::vector<int>& dst);

		void setPlotData(std::vector<float>& v1, std::vector<float>& v2, std::vector<cv::Point2d>& data);
		void setPlotData(std::vector<double>& v1, std::vector<double>& v2, std::vector<cv::Point2d>& data);

		cp::Plot pt;
		std::vector<cv::Point2d> plotsRAW;
		std::vector<cv::Point2d> plotsRANK;
		std::vector<int> refRank;
		std::vector<int> srcRank;
		
		template<typename T>
		struct SpearmanOrder
		{
			T data;
			int order;
		};
		std::vector<SpearmanOrder<float>> sporder32f;
		std::vector<SpearmanOrder<double>> sporder64f;
	public:
		void setReference(std::vector<float>& ref);
		void setReference(std::vector<double>& ref);
		double spearmanUsingReference(std::vector<float>& v1);//compute SROCC (vector<float>). 
		double spearmanUsingReference(std::vector<double>& v1);//compute SROCC (vector<float>). 
		double spearman(std::vector<float> v1, std::vector<float> v2, const bool ignoreTie = false, const bool isPlot = false);//compute SROCC (vector<float>). 
		double spearman(std::vector<double> v1, std::vector<double> v2, const bool ignoreTie = false, const bool isPlot = false);//compute SROCC ((vector<double>)). 
		void plot(const bool isWait = true, const double rawMin = 0.0, const double rawMax = 0.0);
	};
}
