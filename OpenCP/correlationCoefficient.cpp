#include "correlationCoefficient.hpp"
#include "inlineSIMDFunctions.hpp"
#include <numeric>
#include <timer.hpp>
using namespace cv;
using namespace std;

#include "x86simdsort/avx2-32bit-half.hpp"
#include "x86simdsort/avx2-32bit-qsort.hpp"
#include "x86simdsort/avx2-64bit-qsort.hpp"
#include "x86simdsort/avx2-emu-funcs.hpp"

#include "x86simdsort/avx512-64bit-common.h"
#include "x86simdsort/avx512-32bit-qsort.hpp"

#include "x86simdsort/xss-common-argsort.h"

namespace cp
{
#pragma region Pearson
	template<typename T>
	double PearsonCorrelationCoefficient::mean(vector<T>& s)
	{
		const double sum = accumulate(s.begin(), s.end(), 0.0);
		return sum / double(s.size());
	}

	template<typename T>
	double PearsonCorrelationCoefficient::covariance(vector<T>& s1, vector<T>& s2)
	{
		CV_Assert(s1.size() == s2.size());

		const double mean1 = mean(s1), mean2 = mean(s2);
		double sum = (double(s1[0]) - mean1) * (double(s2[0]) - mean2);
		for (unsigned int i = 1; i < s1.size(); i++)
		{
			sum += (double(s1[i]) - mean1) * (double(s2[i]) - mean2);
		}

		return double(sum) / double(s1.size());
	}

	template<typename T>
	double PearsonCorrelationCoefficient::stddev(vector<T>& v1)
	{
		return sqrt(covariance(v1, v1));
	}

	double PearsonCorrelationCoefficient::compute(vector<int>& v1, vector<int>& v2)
	{
		const double v = stddev(v1) * stddev(v2);
		if (v == 0) return -1.0; // negative
		else return covariance(v1, v2) / v;
	}

	double PearsonCorrelationCoefficient::compute(vector<float>& v1, vector<float>& v2)
	{
		const double v = stddev(v1) * stddev(v2);
		if (v == 0) return -1.0; // negative
		else return covariance(v1, v2) / v;
	}

	double PearsonCorrelationCoefficient::compute(vector<double>& v1, vector<double>& v2)
	{
		const double v = stddev(v1) * stddev(v2);
		if (v == 0) return -1.0; // negative
		else return covariance(v1, v2) / v;
	}
#pragma endregion

#pragma region Kendall
	template<typename T>
	double KendallRankOrderCorrelationCoefficient::body(vector<T>& v1, vector<T>& v2)
	{
		CV_Assert(v1.size() == v2.size());
		const int N = (unsigned int)v1.size();
		if (N == 0) cout << "empty vector: KendallRankOrderCorrelationCoefficient" << endl;

		int nC = 0, nD = 0;
		const double n0 = double(N) * double(N - 1) * 0.5;
		int n1 = 0, n2 = 0;
		//#pragma omp parallel for reduction(+:nC, nD) 
		for (int i = 0; i < (N - 1); i++)
		{
			for (int j = i + 1; j < N; j++)
			{
				bool flag = false;
				if (v1[i] == v1[j])
				{
					++n1;
					flag = true;
				}
				if (v2[i] == v2[j])
				{
					++n2;
					flag = true;
				}
				if (flag) continue;

				if (v1[i] > v1[j])
				{
					if (v2[i] > v2[j])nC++;
					else nD++;
				}

				if (v1[i] < v1[j])
				{
					if (v2[i] < v2[j])nC++;
					else nD++;
				}

				/*if ((v1[i] > v1[j] && v2[i] > v2[j]) ||
					(v1[i] < v1[j] && v2[i] < v2[j]))
					nC++;
				if ((v1[i] > v1[j] && v2[i] < v2[j]) ||
					(v1[i] < v1[j] && v2[i] > v2[j]))
					nD++;
					if (v1[i] == v1[j])++n1;
				if (v2[i] == v2[j]) ++n2;
					*/
			}
		}
		const double div = (n0 - n1) * (n0 - n2);
		const double tau = (double)(nC - nD) / sqrt(div);
		//cout<<format("\nn0=%d n1=%d n2=%d C=%d D=%d div=%d tau=%f\n", n0, n1, n2, nC, nD, div, tau)<<endl;
		return tau;
	}

	double KendallRankOrderCorrelationCoefficient::compute(vector<int>& x, vector<int>& y)
	{
		return body<int>(x, y);
	}

	double KendallRankOrderCorrelationCoefficient::compute(vector<float>& x, vector<float>& y)
	{
		return body<float>(x, y);
	}

	double KendallRankOrderCorrelationCoefficient::compute(vector<double>& x, vector<double>& y)
	{
		return body<double>(x, y);
	}
#pragma endregion

#pragma region Spearman

	SpearmanRankOrderCorrelationCoefficient::SpearmanRankOrderCorrelationCoefficient()
	{
		;
	}
#pragma region rankTransform
	template<>
	double SpearmanRankOrderCorrelationCoefficient::rankTransformUsingAverageTieScore<int>(vector<int>& src, vector<float>& dst)
	{
		const int n = (int)src.size();
		if (dst.size() != n)dst.resize(n);
		if (sporder32i.size() != n + 1)sporder32i.resize(n + 1);
		sporder32i[n].data = INT_MIN;

		for (int i = 0; i < n; i++)
		{
			sporder32i[i].data = src[i];
			sporder32i[i].order = i;
		}
		sort(sporder32i.begin(), sporder32i.end(), [](const SpearmanOrder<int>& ls, const SpearmanOrder<int>& rs) {return ls.data < rs.data; });//ascending order

		double Tie = 0.0;
		for (int i = 0; i < n;)
		{
			int pre = sporder32i[i].data;
			int j = 1;
			float ave_rank = float(i);
			for (; j < n - i; j++)
			{
				if (pre != sporder32i[i + j].data) break;
				ave_rank += float(i + j);
			}
			ave_rank /= j;
			Tie += pow(j, 3.0) - double(j);
			for (int k = 0; k < j; k++)
			{
				//cout << dst[i + k] << "," << ave_rank << endl;
				dst[sporder32i[i + k].order] = ave_rank;
			}
			i += j;
		}

		return Tie / 12.0;
	}

	template<>
	double SpearmanRankOrderCorrelationCoefficient::rankTransformUsingAverageTieScore<float>(vector<float>& src, vector<float>& dst)
	{
		const int n = (int)src.size();
		if (dst.size() != n)dst.resize(n);
		if (sporder32f.size() != n + 1)sporder32f.resize(n + 1);
		sporder32f[n].data = FLT_MIN;

		for (int i = 0; i < n; i++)
		{
			sporder32f[i].data = src[i];
			sporder32f[i].order = i;
		}
		sort(sporder32f.begin(), sporder32f.end(), [](const SpearmanOrder<float>& ls, const SpearmanOrder<float>& rs) {return ls.data < rs.data; });//ascending order

		double Tie = 0.0;
		for (int i = 0; i < n;)
		{
			float pre = sporder32f[i].data;
			int j = 1;
			float ave_rank = float(i);
			for (; j < n - i; j++)
			{
				if (pre != sporder32f[i + j].data) break;
				ave_rank += float(i + j);
			}
			ave_rank /= j;
			Tie += pow(j, 3.0) - double(j);
			for (int k = 0; k < j; k++)
			{
				//cout << dst[i + k] << "," << ave_rank << endl;
				dst[sporder32f[i + k].order] = ave_rank;
			}
			i += j;
		}

		return Tie / 12.0;
	}

	template<>
	double SpearmanRankOrderCorrelationCoefficient::rankTransformUsingAverageTieScore<double>(vector<double>& src, vector<float>& dst)
	{
		const int n = (int)src.size();
		if (dst.size() != n)dst.resize(n);
		if (sporder64f.size() != n + 1)sporder64f.resize(n + 1);
		sporder64f[n].data = DBL_MIN;//boundary

		for (int i = 0; i < n; i++)
		{
			sporder64f[i].data = src[i];
			sporder64f[i].order = i;
		}
		sort(sporder64f.begin(), sporder64f.end(), [](const SpearmanOrder<double>& ls, const SpearmanOrder<double>& rs) {return ls.data < rs.data; });//ascending order
		double Tie = 0.0;
		for (int i = 0; i < n;)
		{
			double pre = sporder64f[i].data;
			int j = 1;
			float ave_rank = float(i);
			for (; j < n; j++)
			{
				if (pre != sporder64f[i + j].data) break;
				ave_rank += float(i + j);
			}
			ave_rank /= j;
			Tie += pow(j, 3.0) - double(j);
			for (int k = 0; k < j; k++)
			{
				//cout << dst[i + k] << "," << ave_rank << endl;
				dst[sporder64f[i + k].order] = ave_rank;
			}
			i += j;
		}

		return Tie / 12.0;
	}


	template<>
	void SpearmanRankOrderCorrelationCoefficient::rankTransformIgnoreTie<int>(vector<int>& src, vector<float>& dst)
	{
		const int n = (int)src.size();
		if (dst.size() != n)dst.resize(n);
		if (sporder32i.size() != n)sporder32i.resize(n);
		for (int i = 0; i < n; i++)
		{
			sporder32i[i].data = src[i];
			sporder32i[i].order = i;
		}

		sort(sporder32i.begin(), sporder32i.end(), [](const SpearmanOrder<int>& ls, const SpearmanOrder<int>& rs) {return ls.data < rs.data; });//ascending order
		for (int i = 0; i < n; i++)
		{
			dst[sporder32i[i].order] = (float)i;
		}
	}

	//{
	//	// データのポインタを使ってソートする関数
	//	template <typename T>
	//	std::vector<std::pair<size_t, const T*>> sort_data_with_indices_shallow(const std::vector<T>& data) {
	//		// データのポインタとインデックスをペアにしたベクトルを作成
	//		

	//		for (size_t i = 0; i < data.size(); ++i) {
	//			indexed_data[i] = { i, &data[i] }; // ポインタをコピー（シャローコピー）
	//		}

	//		// データのポインタを基にペアをソート
	//		std::sort(indexed_data.begin(), indexed_data.end(),
	//			[](const std::pair<size_t, const T*>& a, const std::pair<size_t, const T*>& b) {
	//				return *(a.second) < *(b.second); // データ自体を比較するためにポインタをデリファレンス
	//			});

	//		return indexed_data;
	//	}

	//}

	template<>
	void SpearmanRankOrderCorrelationCoefficient::rankTransformIgnoreTie<float>(vector<float>& src, vector<float>& dst)
	{
		const int n = (int)src.size();
		if (dst.size() != n) dst.resize(n);

		if (false)
		{
			if (sporder32f.size() != n) sporder32f.resize(n);
			for (int i = 0; i < n; i++)
			{
				sporder32f[i].data = src[i];
				sporder32f[i].order = i;
			}
			std::sort(sporder32f.begin(), sporder32f.end(), [](const SpearmanOrder<float>& ls, const SpearmanOrder<float>& rs) {return ls.data < rs.data; });//ascending order
			for (int i = 0; i < n; i++)
			{
				dst[sporder32f[i].order] = (float)i;
			}
		}
		else
		{
			if (indices.size() != n) indices.resize(src.size());
			std::iota(indices.begin(), indices.end(), 0);
			avx2_argsort(src.data(), indices.data(), src.size(), false);
			//avx512_argsort(src.data(), indices.data(), src.size(), false);
			for (int i = 0; i < n; i++)
			{
				dst[indices[i]] = (float)i;
			}
		}
	}

	template<>
	void SpearmanRankOrderCorrelationCoefficient::rankTransformIgnoreTie<double>(vector<double>& src, vector<float>& dst)
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
			dst[sporder64f[i].order] = (float)i;
		}
	}

	template<typename T>
	void SpearmanRankOrderCorrelationCoefficient::rankTransformBruteForce(std::vector<T>& ns, std::vector<float>& rs)
	{
		//Ignore tie store
		const unsigned int n = (unsigned int)ns.size();
		rs.resize(n);

		for (int i = 0; i < n; i++)
		{
			int c = 0;
			for (int j = 0; j < n; j++)
			{
				if (ns[i] < ns[j]) c++;
			}
			rs[i] = c + 1;
		}
	}
#pragma endregion

#pragma region compute
	template<typename T>
	double SpearmanRankOrderCorrelationCoefficient::computeRankDifference(std::vector<T>& Rsrc, std::vector<T>& Rref)
	{
		const int n = (int)Rsrc.size();
		double ret = 0.0;
		if (typeid(T) == typeid(float))
		{
			const int N = get_simd_floor(n, 8);
			__m256 msum = _mm256_setzero_ps();
			const float* s = &Rsrc[0];
			const float* r = &Rref[0];
			for (int i = 0; i < N; i += 8)
			{
				__m256 msub = _mm256_sub_ps(_mm256_loadu_ps(s + i), _mm256_loadu_ps(r + i));
				msum = _mm256_fmadd_ps(msub, msub, msum);
			}
			ret = _mm256_reduceadd_ps(msum);
			for (int i = N; i < n; i++)
			{
				const float sub = Rsrc[i] - Rref[i];
				ret += sub * sub;
			}
		}
		else
		{
			for (int i = 0; i < n; i++)
			{
				const double sub = Rsrc[i] - Rref[i];
				ret += sub * sub;
			}
		}
		return ret;
	}

	template<typename T>
	double SpearmanRankOrderCorrelationCoefficient::spearman_(vector<T>& v1, vector<T>& v2, const bool ignoreTie, const bool isPlot, const int plotIndex)
	{
		CV_Assert(v1.size() == v2.size());

		const int n = (int)v1.size();

		if (ignoreTie)
		{
			vector<float> r1(n);
			vector<float> r2(n);
			rankTransformIgnoreTie(v1, r1);
			rankTransformIgnoreTie(v2, r2);

			if (isPlot)
			{
				setPlotData(r1, r2, plotsRANK[plotIndex]);
			}

			const double D2 = computeRankDifference<float>(r1, r2);
			return 1.0 - 6.0 * D2 / (pow(n, 3.0) - (double)n);
		}
		else
		{
			vector<float> r1(n);
			vector<float> r2(n);
			const double T1 = rankTransformUsingAverageTieScore(v1, r1);
			const double T2 = rankTransformUsingAverageTieScore(v2, r2);

			//cout << "T1 " << T1 << ", T2 " << T2 << endl;
			if (isPlot)
			{
				setPlotData(r1, r2, plotsRANK[plotIndex]);
			}

			const bool isUsePearson = false;//for debug
			if (isUsePearson)
			{
				PearsonCorrelationCoefficient pcc;
				return pcc.compute(r1, r2);
			}
			else
			{
				const double D2 = computeRankDifference<float>(r1, r2);
				const double nnn = (pow(n, 3.0) - (double)n) / 6.0;
				return (nnn - D2 - T1 - T2) / (sqrt((nnn - 2.0 * T1) * (nnn - 2.0 * T2)));
			}
		}
	}


	double SpearmanRankOrderCorrelationCoefficient::compute(vector<int>& v1, vector<int>& v2, const bool ignoreTie, const bool isPlot)
	{
		if (isPlot)
		{
			plotsRAW.resize(1);
			setPlotData(v1, v2, plotsRAW[0]);

			plotsRANK.resize(1);
		}
		return spearman_<int>(v1, v2, ignoreTie, isPlot, 0);
	}

	vector<double> SpearmanRankOrderCorrelationCoefficient::compute(vector<vector<int>>& v1, vector<vector<int>>& v2, const bool ignoreTie, const bool isPlot)
	{
		const int size = (int)v1.size();
		vector<double> ret(size + 1);
		if (isPlot)
		{
			int dsize = 0;
			plotsRAW.resize(size);
			for (int i = 0; i < size; i++)
			{
				setPlotData(v1[i], v2[i], plotsRAW[i]);
				ret[i] = spearman_<int>(v1[i], v2[i], ignoreTie, isPlot, i);
				dsize += (int)v1[i].size();
			}
			vector<int> s1, s2;
			for (int i = 0; i < size; i++)
			{
				for (int j = 0; j < v1[i].size(); j++)
				{
					s1.push_back(v1[i][j]);
					s2.push_back(v2[i][j]);
				}
			}
			ret[size] = spearman_<int>(s1, s2, ignoreTie, false, 0);
		}
		return ret;
	}

	double SpearmanRankOrderCorrelationCoefficient::compute(vector<float>& v1, vector<float>& v2, const bool ignoreTie, const bool isPlot)
	{
		if (isPlot)
		{
			plotsRAW.resize(1);
			setPlotData(v1, v2, plotsRAW[0]);

			plotsRANK.resize(1);
		}
		return spearman_<float>(v1, v2, ignoreTie, isPlot, 0);
	}

	vector<double> SpearmanRankOrderCorrelationCoefficient::compute(vector<vector<float>>& v1, vector<vector<float>>& v2, const bool ignoreTie, const bool isPlot)
	{
		const int size = (int)v1.size();
		vector<double> ret(size + 1);
		if (isPlot)
		{
			int dsize = 0;
			plotsRAW.resize(size);
			plotsRANK.resize(size);
			for (int i = 0; i < size; i++)
			{
				setPlotData(v1[i], v2[i], plotsRAW[i]);
				ret[i] = spearman_<float>(v1[i], v2[i], ignoreTie, isPlot, i);
				dsize += (int)v1[i].size();
			}
			vector<float> s1, s2;
			for (int i = 0; i < size; i++)
			{
				for (int j = 0; j < v1[i].size(); j++)
				{
					s1.push_back(v1[i][j]);
					s2.push_back(v2[i][j]);
				}
			}
			ret[size] = spearman_<float>(s1, s2, ignoreTie, false, 0);
		}
		return ret;
	}

	double SpearmanRankOrderCorrelationCoefficient::compute(vector<double>& v1, vector<double>& v2, const bool ignoreTie, const bool isPlot)
	{
		if (isPlot)
		{
			plotsRAW.resize(1);
			setPlotData(v1, v2, plotsRAW[0]);

			plotsRANK.resize(1);
		}
		return spearman_<double>(v1, v2, ignoreTie, isPlot, 0);
	}

	vector<double> SpearmanRankOrderCorrelationCoefficient::compute(vector<vector<double>>& v1, vector<vector<double>>& v2, const bool ignoreTie, const bool isPlot)
	{
		const int size = (int)v1.size();
		vector<double> ret(size + 1);
		if (isPlot)
		{
			int dsize = 0;
			plotsRAW.resize(size);
			for (int i = 0; i < size; i++)
			{
				setPlotData(v1[i], v2[i], plotsRAW[i]);
				ret[i] = spearman_<double>(v1[i], v2[i], ignoreTie, isPlot, i);
				dsize += (int)v1[i].size();
			}
			vector<double> s1, s2;
			for (int i = 0; i < size; i++)
			{
				for (int j = 0; j < v1[i].size(); j++)
				{
					s1.push_back(v1[i][j]);
					s2.push_back(v2[i][j]);
				}
			}
			ret[size] = spearman_<double>(s1, s2, ignoreTie, false, 0);
		}
		return ret;
	}
#pragma endregion

#pragma region set
	void SpearmanRankOrderCorrelationCoefficient::setReference(std::vector<int>& ref, const bool isIgnoreTie)
	{
		if (isIgnoreTie)
		{
			rankTransformIgnoreTie<int>(ref, refRank);
			Tref = 0.0;
		}
		else
		{
			Tref = rankTransformUsingAverageTieScore<int>(ref, refRank);
		}
	}

	void SpearmanRankOrderCorrelationCoefficient::setReference(std::vector<float>& ref, const bool isIgnoreTie)
	{
		if (isIgnoreTie)
		{
			rankTransformIgnoreTie<float>(ref, refRank);
			Tref = 0.0;
		}
		else
		{
			Tref = rankTransformUsingAverageTieScore<float>(ref, refRank);
		}
	}

	void SpearmanRankOrderCorrelationCoefficient::setReference(std::vector<double>& ref, const bool isIgnoreTie)
	{
		if (isIgnoreTie)
		{
			rankTransformIgnoreTie<double>(ref, refRank);
			Tref = 0.0;
		}
		else
		{
			Tref = rankTransformUsingAverageTieScore<double>(ref, refRank);
		}
	}
#pragma endregion

	double SpearmanRankOrderCorrelationCoefficient::computeUsingReference(std::vector<int>& src, const bool isIgnoreTie)
	{
		const int n = (int)src.size();
		double Tsrc = 0.0;
		if (isIgnoreTie) rankTransformIgnoreTie(src, srcRank);
		else Tsrc = rankTransformUsingAverageTieScore(src, srcRank);

		const double D2 = computeRankDifference<float>(srcRank, refRank);
		const double nnn = (pow(n, 3.0) - (double)n) / 6.0;
		//cout << "Tsrc " << Tsrc << ", Tref " << Tref << endl;
		return (nnn - D2 - Tsrc - Tref) / (sqrt((nnn - 2.0 * Tsrc) * (nnn - 2.0 * Tref)));
	}

	double SpearmanRankOrderCorrelationCoefficient::computeUsingReference(std::vector<float>& src, const bool isIgnoreTie)
	{
		const int n = (int)src.size();
		double Tsrc = 0.0;
		if (isIgnoreTie) rankTransformIgnoreTie(src, srcRank);
		else Tsrc = rankTransformUsingAverageTieScore(src, srcRank);

		const double D2 = computeRankDifference<float>(srcRank, refRank);
		const double nnn = (pow(n, 3.0) - (double)n) / 6.0;
		//cout <<"Tsrc "<<Tsrc << ", Tref " << Tref << endl;
		return (nnn - D2 - Tsrc - Tref) / (sqrt((nnn - 2.0 * Tsrc) * (nnn - 2.0 * Tref)));
	}

	double SpearmanRankOrderCorrelationCoefficient::computeUsingReference(std::vector<double>& src, const bool isIgnoreTie)
	{
		const int n = (int)src.size();
		double Tsrc = 0.0;
		if (isIgnoreTie) rankTransformIgnoreTie(src, srcRank);
		else Tsrc = rankTransformUsingAverageTieScore(src, srcRank);

		const double D2 = computeRankDifference<float>(srcRank, refRank);
		const double nnn = (pow(n, 3.0) - (double)n) / 6.0;
		//cout << "Tsrc " << Tsrc << ", Tref " << Tref << endl;
		return (nnn - D2 - Tsrc - Tref) / (sqrt((nnn - 2.0 * Tsrc) * (nnn - 2.0 * Tref)));
	}

#pragma region plot
	template<typename T>
	void SpearmanRankOrderCorrelationCoefficient::setPlotData(const vector<T>& v1, const vector<T>& v2, vector<Point2d>& data)
	{
		//cout << v1.size() << "," << v2.size() << endl;
		data.resize(v1.size());
		for (int i = 0; i < v1.size(); i++)
		{
			data[i] = Point2d(v1[i], v2[i]);
		}
	}

	void SpearmanRankOrderCorrelationCoefficient::plot(const bool isWait, const double rawMin, const double rawMax, vector<string> labels)
	{
		if (labels.size() == 0) pt.setKey(Plot::KEY::NOKEY);
		else
		{
			if (labels.size() == plotsRAW.size())
			{
				for (int i = 0; i < plotsRAW.size(); i++)
				{
					pt.setPlotTitle(i, labels[i]);
					if (i == 0) pt.setPlotColor(i, COLOR_RED);
					if (i == 1) pt.setPlotColor(i, COLOR_GREEN);
					if (i == 2) pt.setPlotColor(i, COLOR_BLUE);
				}
			}
		}

		for (int i = 0; i < plotsRAW.size(); i++)
		{
			pt.setPlotLineType(i, Plot::LINE::NOLINE);
			pt.push_back(plotsRAW[i], i);
		}
		if (rawMin != 0.0 || rawMax != 0.0) pt.setYRange(rawMin, rawMax);
		pt.plot("SROCC-RAW", isWait);
		pt.clear();
		pt.unsetXYRange();
		for (int i = 0; i < plotsRAW.size(); i++)
		{
			pt.push_back(plotsRANK[i], i);
		}
		pt.plot("SROCC-RANK", isWait);
		pt.clear();
	}

	void SpearmanRankOrderCorrelationCoefficient::plotwithAdditionalPoints(const vector<vector<Point2d>>& additionalPoints, const bool isWait, const double rawMin, const double rawMax, vector<string> labels, vector<string> xylabels)
	{
		if (labels.size() == 0) pt.setKey(Plot::KEY::NOKEY);
		else
		{
			if (labels.size() == plotsRAW.size())
			{
				for (int i = 0; i < plotsRAW.size(); i++)
				{
					pt.setPlotTitle(i, labels[i]);
					if (i == 0) pt.setPlotColor(i, COLOR_RED);
					if (i == 1) pt.setPlotColor(i, COLOR_GREEN);
					if (i == 2) pt.setPlotColor(i, COLOR_BLUE);
				}
			}
		}

		pt.setKey(Plot::KEY::NOKEY);
		if (xylabels.size() == 0)
		{
			pt.setXLabel("SROCC-MOS");
			pt.setYLabel("SROCC-SCORE");
		}
		else
		{
			pt.setXLabel("SROCC-" + xylabels[0]);
			pt.setYLabel("SROCC-" + xylabels[1]);
		}
		vector<vector<double>> error(additionalPoints.size());
		vector<vector<int>> argmin(additionalPoints.size());
		for (int ai = 0; ai < additionalPoints.size(); ai++)
		{
			error[ai].resize(additionalPoints[ai].size());
			argmin[ai].resize(additionalPoints[ai].size());
			for (int a = 0; a < additionalPoints[ai].size(); a++)
			{
				error[ai][a] = DBL_MAX;
				argmin[ai][a] = 0;
			}
		}
		for (int i = 0; i < plotsRAW.size(); i++)
		{
			pt.setPlotLineType(i, Plot::LINE::NOLINE);
			pt.push_back(plotsRAW[i], i);

			for (int ai = 0; ai < additionalPoints.size(); ai++)
			{
				for (int a = 0; a < additionalPoints[ai].size(); a++)
				{
					for (int j = 0; j < plotsRAW[i].size(); j++)
					{
						double dist = pow(plotsRAW[i][j].x - additionalPoints[ai][a].x, 2.0) + pow(plotsRAW[i][j].y - additionalPoints[ai][a].y, 2.0);
						if (dist < error[ai][a])
						{
							error[ai][a] = dist;
							argmin[ai][a] = j;
						}
					}
				}
			}

			for (int ai = 0; ai < additionalPoints.size(); ai++)
			{
				const int additionalIndex = (int)plotsRAW.size() + ai;
				Scalar color = ai == 0 ? COLOR_RED : (ai == 1) ? COLOR_GREEN : COLOR_BLUE;
				for (int a = 0; a < additionalPoints[ai].size(); a++)
				{
					pt.setPlotLineType(additionalIndex, Plot::LINE::NOLINE);
					pt.setPlotSymbol(additionalIndex, 4);
					pt.setPlotColor(additionalIndex, color);
					pt.setPlotLineWidth(additionalIndex, 4);
					pt.push_back(additionalPoints[ai][a].x, additionalPoints[ai][a].y, additionalIndex);
				}
			}
		}

		//if (rawMin != 0.0 || rawMax != 0.0) pt.setYRange(rawMin, rawMax);
		pt.plot("SROCC-RAW", isWait);
		pt.clear();
		if (xylabels.size() == 0)
		{
			pt.setYLabel("SROCC-RANK-MOS");
			pt.setYLabel("SROCC-RANK-SCORE");
		}
		else
		{
			pt.setXLabel("SROCC-RANK-" + xylabels[0]);
			pt.setYLabel("SROCC-RANK-" + xylabels[1]);
		}
		pt.unsetXYRange();
		for (int i = 0; i < plotsRAW.size(); i++)
		{
			pt.push_back(plotsRANK[i], i);
			for (int ai = 0; ai < additionalPoints.size(); ai++)
			{
				const int additionalIndex = (int)plotsRAW.size() + ai;
				Scalar color = ai == 0 ? COLOR_RED : (ai == 1) ? COLOR_GREEN : COLOR_BLUE;
				for (int a = 0; a < additionalPoints[ai].size(); a++)
				{
					pt.setPlotLineType(additionalIndex, Plot::LINE::NOLINE);
					pt.setPlotSymbol(additionalIndex, 4);
					pt.setPlotColor(additionalIndex, color);
					pt.setPlotLineWidth(additionalIndex, 4);
					pt.push_back(plotsRANK[i][argmin[ai][a]].x, plotsRANK[i][argmin[ai][a]].y, additionalIndex);
				}
			}
		}

		pt.plot("SROCC-RANK", isWait);
		pt.clear();
	}

	void SpearmanRankOrderCorrelationCoefficient::plotwithAdditionalPoints(const vector<Point2d>& additionalPoints, const bool isWait, const double rawMin, const double rawMax, vector<string> labels, vector<string> xylabels)
	{
		vector<vector<Point2d>> vv;
		vv.push_back(additionalPoints);
		plotwithAdditionalPoints(vv, isWait, rawMin, rawMax, labels, xylabels);
	}

	void SpearmanRankOrderCorrelationCoefficient::plotwithAdditionalPoints(const Point2d& additionalPoints, const bool isWait, const double rawMin, const double rawMax, vector<string> labels, vector<string> xylabels)
	{
		vector<vector<Point2d>> vv;
		vector<Point2d> v;
		v.push_back(additionalPoints);
		vv.push_back(v);
		plotwithAdditionalPoints(vv, isWait, rawMin, rawMax, labels, xylabels);
	}
#pragma endregion
#pragma endregion
}