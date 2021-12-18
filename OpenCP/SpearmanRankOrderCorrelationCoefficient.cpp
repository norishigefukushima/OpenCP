#include "SpearmanRankOrderCorrelationCoefficient.hpp"
#include <numeric>

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
			double sum = 0;
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
			dest.push_back(R[nPosition - 1]);
		}
	}

	template<typename T>
	double SpearmanRankOrderCorrelationCoefficient::spearman_(vector<T>& v1, vector<T>& v2)
	{
		CV_Assert(v1.size() == v2.size());

		const int n = v1.size();
		vector<T> R1;
		vector<T> R2;
		vector<T> d;
		vector<T> vector1(v1); // original vector v1
		vector<T> vector2(v2); // original vector v2

		sort(v1.begin(), v1.end());//ascending order
		sort(v2.begin(), v2.end());

		Rank(v1, vector1, R1);
		Rank(v2, vector2, R2);

		//Method 1:  Pearson correlation coefficient, but applied to the rank variables.
		//cout << "Spearman correlation = " << pearson(R1,R2) <<endl;
		setPlotData(R1, R2, plotsRANK);
		return pearson(R1, R2);

		//Method 2 : Use the spearman correlation formular( Only if all n ranks are distinct integers)
		for (int k = 0; k < n; k++)
		{
			double diff = R1[k] - R2[k]; // Difference d where R1.size() = R2.size()
			double sq_diff = pow(diff, 2);
			d.push_back(sq_diff);
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

	void SpearmanRankOrderCorrelationCoefficient::plot(const bool isWait)
	{
		pt.setPlotLineType(0, Plot::LINE::NOLINE);

		pt.push_back(plotsRAW);
		pt.plot("ROCC-RAW", isWait);
		pt.clear();
		pt.push_back(plotsRANK);
		pt.plot("ROCC-RANK", isWait);
		pt.clear();
	}

	double SpearmanRankOrderCorrelationCoefficient::spearman(vector<float>& v1, vector<float>& v2)
	{
		setPlotData(v1, v2, plotsRAW);
		return spearman_<float>(v1, v2);
	}

	double SpearmanRankOrderCorrelationCoefficient::spearman(vector<double>& v1, vector<double>& v2)
	{
		setPlotData(v1, v2, plotsRAW);
		return spearman_<double>(v1, v2);
	}
}