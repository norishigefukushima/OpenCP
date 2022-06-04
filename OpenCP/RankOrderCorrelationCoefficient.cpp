#include "RankOrderCorrelationCoefficient.hpp"
#include <numeric>
#include <timer.hpp>

using namespace cv;
using namespace std;

namespace cp
{
	template<typename T>
	double SpearmanRankOrderCorrelationCoefficient::mean(vector<T>& s)
	{
		double sum = accumulate(s.begin(), s.end(), 0.0);
		return sum / double(s.size());
	}

	template<typename T>
	double SpearmanRankOrderCorrelationCoefficient::covariance(vector<T>& s1, vector<T>& s2)
	{
		CV_Assert(s1.size() == s2.size());

		double mean1 = mean(s1), mean2 = mean(s2);
		double sum = (double(s1[0]) - mean1) * (double(s2[0]) - mean2);
		for (unsigned int i = 1; i < s1.size(); i++)
		{
			sum += (double(s1[i]) - mean1) * (double(s2[i]) - mean2);
		}

		return double(sum) / double(s1.size());
	}

	template<typename T>
	double SpearmanRankOrderCorrelationCoefficient::std_dev(vector<T>& v1)
	{
		return sqrt(covariance(v1, v1));
	}

	template<typename T>
	double SpearmanRankOrderCorrelationCoefficient::pearson(vector<T>& v1, vector<T>& v2)
	{
		const double v = std_dev(v1) * std_dev(v2);
		if (v == 0) return -1; // negative
		else return covariance(v1, v2) / v;
	}

	template<typename T>
	void SpearmanRankOrderCorrelationCoefficient::searchList(vector<T>& theArray, int sizeOfTheArray, double findFor, vector<int>& index)
	{
		vector<int> foundIndices;

		int j = 0;
		for (int i = 0; i < sizeOfTheArray; i++)
		{
			if (theArray[i] == findFor)
			{
				foundIndices.push_back(i);
				j++;
			}
		}

		if (foundIndices.size() != 0)
		{
			//cout << " Found in index: ";
			for (int i = 0; i < foundIndices.size(); i++)
			{
				// cout << foundIndices[i]+1 << " ";
				index.push_back(foundIndices[i] + 1);
			}
		}
		else
		{
			// cout << " Not found in array";
		}
	}

	template<typename T>
	void SpearmanRankOrderCorrelationCoefficient::Rank(vector<T>& vec, vector<T>& orig_vect, vector<T>& dest)
	{
		assert(vec.size() == orig_vect.size());
		const int size = (int)vec.size();
		vector<double> R(size);
		vector<int> Indices;
		// Declaring new vector and copying element of old vector constructor method, Deep copy
		vector<T> vect2(vec); // vect2 is a sorted list

		// assign rank for Sorted list	
		for (int k = 0; k < size; k++)
		{
			R[k] = k + 1; // 1 start
		}

		vec.resize(std::distance(vec.begin(), std::unique(vec.begin(), vec.end())));

		//Break Ties
		for (int k = 0; k < vec.size(); k++)
		{
			// Search for the index position by value
			Indices.clear();
			searchList(vect2, size, vec[k], Indices);
			// Find mean position
			double sum = 0.0;
			for (int i = 0; i < Indices.size(); i++)
			{
				sum += R[Indices[i] - 1];
			}
			double mean_index = sum / Indices.size();
			//change the rank at ties position
			for (int j = 0; j < Indices.size(); j++)
			{
				R[Indices[j] - 1] = mean_index;
			}
		}

		// Search sorted list for index of item on original vector	
		double nPosition;

		for (int k = 0; k < orig_vect.size(); k++)
		{
			Indices.clear();
			searchList(vect2, size, orig_vect[k], Indices);
			nPosition = Indices[0]; // just need one ocurrence		
			// Get the respective postion in sorted list then pushback in rank		
			if (nPosition == 0)cout << "zero" << endl;
			dest.push_back(T(R[(int)nPosition - 1]));
		}
	}

	
	template<>
	void SpearmanRankOrderCorrelationCoefficient::rankTransformIgnoreTie<float>(vector<float>& src, vector<int>& dst)
	{
		const int n = (int)src.size();
		dst.resize(n);
		sporder32f.resize(n);
		//vector<SpearmanOrder<T>> data(n);
		for (int i = 0; i < n; i++)
		{
			sporder32f[i].data = src[i];
			sporder32f[i].order = i;
		}
		sort(sporder32f.begin(), sporder32f.end(), [](const SpearmanOrder<float>& ls, const SpearmanOrder<float>& rs) {return ls.data < rs.data; });//ascending order
		for (int i = 0; i < n; i++)
		{
			dst[sporder32f[i].order] = i;
		}
	}

	template<>
	void SpearmanRankOrderCorrelationCoefficient::rankTransformIgnoreTie<double>(vector<double>& src, vector<int>& dst)
	{
		const int n = (int)src.size();
		dst.resize(n);
		sporder64f.resize(n);
		//vector<SpearmanOrder<T>> data(n);
		for (int i = 0; i < n; i++)
		{
			sporder64f[i].data = src[i];
			sporder64f[i].order = i;
		}
		sort(sporder64f.begin(), sporder64f.end(), [](const SpearmanOrder<double>& ls, const SpearmanOrder<double>& rs) {return ls.data < rs.data; });//ascending order
		for (int i = 0; i < n; i++)
		{
			dst[sporder64f[i].order] = i;
		}
	}

	template<typename T>
	double SpearmanRankOrderCorrelationCoefficient::spearman_(vector<T>& v1, vector<T>& v2, const bool ignoreTie, const bool isPlot)
	{
		CV_Assert(v1.size() == v2.size());

		const int n = (int)v1.size();
		vector<T> R1;
		vector<T> R2;
		vector<T> d;

		if (ignoreTie)
		{
			vector<SpearmanOrder<T>> data1(n);
			vector<SpearmanOrder<T>> data2(n);
			for (int i = 0; i < n; i++)
			{
				data1[i].data = v1[i];
				data1[i].order = i;
				data2[i].data = v2[i];
				data2[i].order = i;
			}
			sort(data1.begin(), data1.end(), [](const SpearmanOrder<T>& ls, const SpearmanOrder<T>& rs) {return ls.data < rs.data; });//ascending order
			sort(data2.begin(), data2.end(), [](const SpearmanOrder<T>& ls, const SpearmanOrder<T>& rs) {return ls.data < rs.data; });//ascending order
			vector<int> r1(n);
			vector<int> r2(n);
			for (int i = 0; i < n; i++)
			{
				r1[data1[i].order] = i;
				r2[data2[i].order] = i;
			}
			double D2 = 0.0;
			for (int i = 0; i < n; i++)
			{
				//cout << R1[i] << "," << R2[i] << endl;
				const double sub = r1[i] - r2[i];
				D2 += sub * sub;
			}
			return 1.0 - 6.0 * D2 / (pow(n, 3.0) - (double)n);
		}
		else
		{
			vector<T> vector1(v1); // original vector v1
			vector<T> vector2(v2); // original vector v2
			{
				//Timer t("sort");
				sort(v1.begin(), v1.end());//ascending order
				sort(v2.begin(), v2.end());
			}
			//Timer t("Rank");
			Rank(v1, vector1, R1);
			Rank(v2, vector2, R2);
		}
		//Method 1:  Pearson correlation coefficient, but applied to the rank variables.
		//cout << "Spearman correlation = " << pearson(R1,R2) <<endl;
		if (isPlot)setPlotData(R1, R2, plotsRANK);
		double ret = 0.0;
		{
			//Timer t("pearson");
			ret = pearson(R1, R2);
		}
		return ret;

		//Method 2 : Use the spearman correlation formular( Only if all n ranks are distinct integers)
		for (int k = 0; k < n; k++)
		{
			double diff = R1[k] - R2[k]; // Difference d where R1.size() = R2.size()
			double sq_diff = pow(diff, 2);
			d.push_back(T(sq_diff));
		}
		// Sum the Squared difference
		double sum = std::accumulate(d.begin(), d.end(), 0.0);
		int en = n;
		double en3n = (en * en * en) - en;
		double numerator = 6 * sum;
		double corr = 1 - (numerator / en3n);
		//cout << "Spearman correlation  (Method 2 ) = " << corr <<endl;
		//cout << "Note: Method 2: Only if all n ranks are distinct integers, then Method 1 = Method2 ";
	}

	void SpearmanRankOrderCorrelationCoefficient::setPlotData(vector<float>& v1, vector<float>& v2, vector<cv::Point2d>& data)
	{
		data.resize(v1.size());
		for (int i = 0; i < v1.size(); i++)
		{
			data[i] = Point2d(v1[i], v2[i]);
		}
	}

	void SpearmanRankOrderCorrelationCoefficient::setPlotData(vector<double>& v1, vector<double>& v2, vector<cv::Point2d>& data)
	{
		data.resize(v1.size());
		for (int i = 0; i < v1.size(); i++)
		{
			data[i] = Point2d(v1[i], v2[i]);
		}
	}

	void SpearmanRankOrderCorrelationCoefficient::plot(const bool isWait, const double rawMin, const double rawMax)
	{
		pt.setPlotLineType(0, Plot::LINE::NOLINE);
		pt.setKey(Plot::KEY::NOKEY);
		pt.push_back(plotsRAW);
		if (rawMin != 0.0 || rawMax != 0.0) pt.setYRange(rawMin, rawMax);
		pt.plot("SROCC-RAW", isWait);
		pt.clear();
		pt.unsetXYRange();
		pt.push_back(plotsRANK);
		pt.plot("SROCC-RANK", isWait);
		pt.clear();
	}

	double SpearmanRankOrderCorrelationCoefficient::spearman(vector<float> v1, vector<float> v2, const bool ignoreTie, const bool isPlot)
	{
		if (isPlot)setPlotData(v1, v2, plotsRAW);
		return spearman_<float>(v1, v2, ignoreTie, isPlot);
	}

	double SpearmanRankOrderCorrelationCoefficient::spearman(vector<double> v1, vector<double> v2, const bool ignoreTie, const bool isPlot)
	{
		if (isPlot)setPlotData(v1, v2, plotsRAW);
		return spearman_<double>(v1, v2, ignoreTie, isPlot);
	}

	void SpearmanRankOrderCorrelationCoefficient::setReference(std::vector<float>& ref)
	{
		rankTransformIgnoreTie<float>(ref, refRank);
	}

	void SpearmanRankOrderCorrelationCoefficient::setReference(std::vector<double>& ref)
	{
		rankTransformIgnoreTie<double>(ref, refRank);
	}

	double SpearmanRankOrderCorrelationCoefficient::spearmanUsingReference(std::vector<float>& src)
	{
		const int n = (int)src.size();
		rankTransformIgnoreTie(src, srcRank);
		double D2 = 0.0;
		for (int i = 0; i < n; i++)
		{
			const double sub = srcRank[i] - refRank[i];
			D2 += sub * sub;
		}
		return 1.0 - 6.0 * D2 / (pow(n, 3.0) - (double)n);
	}

	double SpearmanRankOrderCorrelationCoefficient::spearmanUsingReference(std::vector<double>& src)
	{
		const int n = (int)src.size();
		rankTransformIgnoreTie(src, srcRank);
		double D2 = 0.0;
		for (int i = 0; i < n; i++)
		{
			const double sub = srcRank[i] - refRank[i];
			D2 += sub * sub;
		}
		return 1.0 - 6.0 * D2 / (pow(n, 3.0) - (double)n);
	}
}