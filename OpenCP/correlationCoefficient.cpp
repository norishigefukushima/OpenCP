#include "correlationCoefficient.hpp"
#include "inlineSIMDFunctions.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
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
	double PearsonCorrelationCoefficient::body(vector<T>& first, vector<T>& second)
	{
		CV_Assert(first.size() == second.size());
		if (first.size() < 2) return std::numeric_limits<double>::quiet_NaN();
		if constexpr (std::is_floating_point_v<T>)
			for (size_t i = 0; i < first.size(); ++i)
				if (!std::isfinite(first[i]) || !std::isfinite(second[i]))
					return std::numeric_limits<double>::quiet_NaN();
		const double inverseCount = 1.0 / static_cast<double>(first.size());
		const double firstMean = std::accumulate(first.begin(), first.end(), 0.0) *
			inverseCount;
		const double secondMean = std::accumulate(second.begin(), second.end(), 0.0) *
			inverseCount;
		double covariance = 0.0;
		double firstVariance = 0.0;
		double secondVariance = 0.0;
		for (size_t i = 0; i < first.size(); ++i)
		{
			const double a = static_cast<double>(first[i]) - firstMean;
			const double b = static_cast<double>(second[i]) - secondMean;
			covariance += a * b;
			firstVariance += a * a;
			secondVariance += b * b;
		}
		const double denominator = std::sqrt(firstVariance * secondVariance);
		return denominator > 0.0 ? covariance / denominator :
			std::numeric_limits<double>::quiet_NaN();
	}

	double PearsonCorrelationCoefficient::compute(vector<int>& v1, vector<int>& v2)
	{
		return body(v1, v2);
	}

	double PearsonCorrelationCoefficient::compute(vector<float>& v1, vector<float>& v2)
	{
		return body(v1, v2);
	}

	double PearsonCorrelationCoefficient::compute(vector<double>& v1, vector<double>& v2)
	{
		return body(v1, v2);
	}
#pragma endregion

#pragma region Kendall
	template<typename T>
	double KendallRankOrderCorrelationCoefficient::body(vector<T>& v1, vector<T>& v2)
	{
		CV_Assert(v1.size() == v2.size());
		const size_t count = v1.size();
		if (count < 2) return std::numeric_limits<double>::quiet_NaN();
		if constexpr (std::is_floating_point_v<T>)
			for (size_t i = 0; i < count; ++i)
				if (!std::isfinite(v1[i]) || !std::isfinite(v2[i]))
					return std::numeric_limits<double>::quiet_NaN();

		std::vector<size_t> order(count);
		std::iota(order.begin(), order.end(), size_t{ 0 });
		avx2_argsort(v1.data(), order.data(), count, false);

		uint64_t tiedFirst = 0;
		for (size_t begin = 0; begin < count;)
		{
			size_t end = begin + 1;
			while (end < count && v1[order[end]] == v1[order[begin]]) ++end;
			const uint64_t size = static_cast<uint64_t>(end - begin);
			tiedFirst += size * (size - 1) / 2;
			begin = end;
		}

		std::vector<size_t> secondOrder(count);
		std::iota(secondOrder.begin(), secondOrder.end(), size_t{ 0 });
		avx2_argsort(v2.data(), secondOrder.data(), count, false);
		std::vector<T> sortedSecond(count);
		for (size_t i = 0; i < count; ++i)
			sortedSecond[i] = v2[secondOrder[i]];
		uint64_t tiedSecond = 0;
		for (size_t begin = 0; begin < count;)
		{
			size_t end = begin + 1;
			while (end < count && sortedSecond[end] == sortedSecond[begin]) ++end;
			const uint64_t size = static_cast<uint64_t>(end - begin);
			tiedSecond += size * (size - 1) / 2;
			begin = end;
		}
		sortedSecond.erase(std::unique(sortedSecond.begin(), sortedSecond.end()),
			sortedSecond.end());

		std::vector<uint64_t> fenwick(sortedSecond.size() + 1, 0);
		const auto lowBit = [](const size_t value) { return value & (~value + 1); };
		const auto prefixSum = [&](size_t index)
		{
			uint64_t sum = 0;
			for (; index > 0; index -= lowBit(index)) sum += fenwick[index];
			return sum;
		};
		const auto add = [&](size_t index)
		{
			for (; index < fenwick.size(); index += lowBit(index)) ++fenwick[index];
		};

		uint64_t concordant = 0;
		uint64_t discordant = 0;
		uint64_t processed = 0;
		for (size_t begin = 0; begin < count;)
		{
			size_t end = begin + 1;
			while (end < count && v1[order[end]] == v1[order[begin]]) ++end;
			for (size_t i = begin; i < end; ++i)
			{
				const size_t rank = static_cast<size_t>(std::lower_bound(
					sortedSecond.begin(), sortedSecond.end(), v2[order[i]]) -
					sortedSecond.begin()) + 1;
				const uint64_t less = prefixSum(rank - 1);
				const uint64_t lessOrEqual = prefixSum(rank);
				concordant += less;
				discordant += processed - lessOrEqual;
			}
			for (size_t i = begin; i < end; ++i)
			{
				const size_t rank = static_cast<size_t>(std::lower_bound(
					sortedSecond.begin(), sortedSecond.end(), v2[order[i]]) -
					sortedSecond.begin()) + 1;
				add(rank);
				++processed;
			}
			begin = end;
		}

		const double n = static_cast<double>(count);
		const double pairs = n * (n - 1.0) * 0.5;
		const double denominator = std::sqrt(
			(pairs - static_cast<double>(tiedFirst)) *
			(pairs - static_cast<double>(tiedSecond)));
		if (!(denominator > 0.0))
			return std::numeric_limits<double>::quiet_NaN();
		return (static_cast<double>(concordant) -
			static_cast<double>(discordant)) / denominator;
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

#pragma region Chatterjee
	template<typename T>
	double ChatterjeeCorrelationCoefficient::body(
		vector<T>& predictor, vector<T>& response)
	{
		CV_Assert(predictor.size() == response.size());
		const size_t count = predictor.size();
		if (count < 2) return std::numeric_limits<double>::quiet_NaN();
		if constexpr (std::is_floating_point_v<T>)
			for (size_t i = 0; i < count; ++i)
				if (!std::isfinite(predictor[i]) || !std::isfinite(response[i]))
					return std::numeric_limits<double>::quiet_NaN();

		std::vector<size_t> predictorOrder(count);
		std::iota(predictorOrder.begin(), predictorOrder.end(), size_t{ 0 });
		avx2_argsort(predictor.data(), predictorOrder.data(), count, false);
		// x86-simd-sort is not stable. Restore original order only inside equal
		// predictor groups so the continuous-X estimator has deterministic ties.
		for (size_t begin = 0; begin < count;)
		{
			size_t end = begin + 1;
			while (end < count && predictor[predictorOrder[end]] ==
				predictor[predictorOrder[begin]]) ++end;
			if (end - begin > 1)
				std::sort(predictorOrder.begin() + begin, predictorOrder.begin() + end);
			begin = end;
		}

		std::vector<size_t> responseOrder(count);
		std::iota(responseOrder.begin(), responseOrder.end(), size_t{ 0 });
		avx2_argsort(response.data(), responseOrder.data(), count, false);
		std::vector<size_t> lessEqualRank(count);
		std::vector<size_t> greaterEqualRank(count);
		for (size_t begin = 0; begin < count;)
		{
			size_t end = begin + 1;
			while (end < count &&
				response[responseOrder[begin]] == response[responseOrder[end]]) ++end;
			for (size_t index = begin; index < end; ++index)
			{
				lessEqualRank[responseOrder[index]] = end;
				greaterEqualRank[responseOrder[index]] = count - begin;
			}
			begin = end;
		}

		double numerator = 0.0;
		for (size_t index = 1; index < count; ++index)
		{
			const long long first = static_cast<long long>(
				lessEqualRank[predictorOrder[index - 1]]);
			const long long second = static_cast<long long>(
				lessEqualRank[predictorOrder[index]]);
			numerator += static_cast<double>(std::llabs(second - first));
		}
		double denominatorSum = 0.0;
		for (const size_t rank : greaterEqualRank)
			denominatorSum += static_cast<double>(rank) *
				static_cast<double>(count - rank);
		const double denominator = 2.0 * denominatorSum;
		if (!(denominator > 0.0))
			return std::numeric_limits<double>::quiet_NaN();
		return 1.0 - static_cast<double>(count) * numerator / denominator;
	}

	double ChatterjeeCorrelationCoefficient::compute(
		vector<int>& predictor, vector<int>& response)
	{
		return body<int>(predictor, response);
	}

	double ChatterjeeCorrelationCoefficient::compute(
		vector<float>& predictor, vector<float>& response)
	{
		return body<float>(predictor, response);
	}

	double ChatterjeeCorrelationCoefficient::compute(
		vector<double>& predictor, vector<double>& response)
	{
		return body<double>(predictor, response);
	}
#pragma endregion

#pragma region CorrelationCoefficientSet
	void CorrelationCoefficientSet::setReference(
		const std::vector<double>& reference)
	{
		if (reference.size() < 2 || !std::all_of(reference.begin(), reference.end(),
			[](const double value) { return std::isfinite(value); }))
			throw std::invalid_argument(
				"CorrelationCoefficientSet reference must contain at least two finite values");
		const size_t count = reference.size();
		const double mean = std::accumulate(reference.begin(), reference.end(), 0.0) /
			static_cast<double>(count);
		referenceCentered.resize(count);
		referenceVariance = 0.0;
		for (size_t i = 0; i < count; ++i)
		{
			referenceCentered[i] = reference[i] - mean;
			referenceVariance += referenceCentered[i] * referenceCentered[i];
		}

		order.resize(count);
		std::iota(order.begin(), order.end(), size_t{ 0 });
		// x86-simd-sort does not modify the reference values.
		avx2_argsort(const_cast<double*>(reference.data()), order.data(), count, false);
		referenceRank.resize(count);
		referenceOrdinalRank.resize(count);
		referenceLessEqualRank.resize(count);
		referenceGreaterEqualRank.resize(count);
		referenceTieScore = 0.0;
		referenceTiedPairs = 0.0;
		chatterjeeDenominator = 0.0;
		size_t ordinal = 0;
		for (size_t begin = 0; begin < count;)
		{
			size_t end = begin + 1;
			while (end < count && reference[order[end]] == reference[order[begin]]) ++end;
			++ordinal;
			const float averageRank = 0.5f * static_cast<float>(begin + end - 1);
			const double size = static_cast<double>(end - begin);
			referenceTieScore += (size * size * size - size) / 12.0;
			referenceTiedPairs += size * (size - 1.0) * 0.5;
			for (size_t index = begin; index < end; ++index)
			{
				const size_t original = order[index];
				referenceRank[original] = averageRank;
				referenceOrdinalRank[original] = ordinal;
				referenceLessEqualRank[original] = end;
				referenceGreaterEqualRank[original] = count - begin;
			}
			begin = end;
		}
		for (const size_t rank : referenceGreaterEqualRank)
			chatterjeeDenominator += static_cast<double>(rank) *
				static_cast<double>(count - rank);
		chatterjeeDenominator *= 2.0;
		fenwick.assign(ordinal + 1, 0);
		predictorRank.resize(count);
	}

	CorrelationCoefficientSet::Result CorrelationCoefficientSet::compute(
		const std::vector<double>& predictor)
	{
		Result result;
		const size_t count = referenceCentered.size();
		if (predictor.size() != count || count < 2 ||
			!std::all_of(predictor.begin(), predictor.end(),
				[](const double value) { return std::isfinite(value); }))
			return result;

		result.pearson = computePearsonUnchecked(predictor);

		order.resize(count);
		std::iota(order.begin(), order.end(), size_t{ 0 });
		avx2_argsort(const_cast<double*>(predictor.data()), order.data(), count, false);
		double predictorTieScore = 0.0;
		double predictorTiedPairs = 0.0;
		for (size_t begin = 0; begin < count;)
		{
			size_t end = begin + 1;
			while (end < count && predictor[order[end]] == predictor[order[begin]]) ++end;
			const float averageRank = 0.5f * static_cast<float>(begin + end - 1);
			const double size = static_cast<double>(end - begin);
			predictorTieScore += (size * size * size - size) / 12.0;
			predictorTiedPairs += size * (size - 1.0) * 0.5;
			for (size_t index = begin; index < end; ++index)
				predictorRank[order[index]] = averageRank;
			// Restore stable input order for Chatterjee xi when predictor ties exist.
			if (end - begin > 1)
				std::sort(order.begin() + begin, order.begin() + end);
			begin = end;
		}

		double squaredRankDifference = 0.0;
		const int n = static_cast<int>(count);
		const int n8 = get_simd_floor(n, 8);
		if (accumulation == Accumulation::Float32)
		{
			constexpr int chunkSize = 256;
			int i = 0;
			for (; i + 31 < n8;)
			{
				const int end = std::min(n8, i + chunkSize);
				__m256 sum0 = _mm256_setzero_ps();
				__m256 sum1 = _mm256_setzero_ps();
				__m256 sum2 = _mm256_setzero_ps();
				__m256 sum3 = _mm256_setzero_ps();
				for (; i + 31 < end; i += 32)
				{
					const __m256 sub0 = _mm256_sub_ps(
						_mm256_loadu_ps(predictorRank.data() + i),
						_mm256_loadu_ps(referenceRank.data() + i));
					const __m256 sub1 = _mm256_sub_ps(
						_mm256_loadu_ps(predictorRank.data() + i + 8),
						_mm256_loadu_ps(referenceRank.data() + i + 8));
					const __m256 sub2 = _mm256_sub_ps(
						_mm256_loadu_ps(predictorRank.data() + i + 16),
						_mm256_loadu_ps(referenceRank.data() + i + 16));
					const __m256 sub3 = _mm256_sub_ps(
						_mm256_loadu_ps(predictorRank.data() + i + 24),
						_mm256_loadu_ps(referenceRank.data() + i + 24));
					sum0 = _mm256_fmadd_ps(sub0, sub0, sum0);
					sum1 = _mm256_fmadd_ps(sub1, sub1, sum1);
					sum2 = _mm256_fmadd_ps(sub2, sub2, sum2);
					sum3 = _mm256_fmadd_ps(sub3, sub3, sum3);
				}
				const __m256 sum = _mm256_add_ps(
					_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
				squaredRankDifference += _mm256_reduceadd_ps(sum);
			}
			__m256 remainder = _mm256_setzero_ps();
			for (; i < n8; i += 8)
			{
				const __m256 sub = _mm256_sub_ps(
					_mm256_loadu_ps(predictorRank.data() + i),
					_mm256_loadu_ps(referenceRank.data() + i));
				remainder = _mm256_fmadd_ps(sub, sub, remainder);
			}
			squaredRankDifference += _mm256_reduceadd_ps(remainder);
		}
		else
		{
			__m256d sumLow = _mm256_setzero_pd();
			__m256d sumHigh = _mm256_setzero_pd();
			for (int i = 0; i < n8; i += 8)
			{
				const __m256 source = _mm256_loadu_ps(predictorRank.data() + i);
				const __m256 reference = _mm256_loadu_ps(referenceRank.data() + i);
				const __m256d low = _mm256_sub_pd(
					_mm256_cvtps_pd(_mm256_castps256_ps128(source)),
					_mm256_cvtps_pd(_mm256_castps256_ps128(reference)));
				const __m256d high = _mm256_sub_pd(
					_mm256_cvtps_pd(_mm256_extractf128_ps(source, 1)),
					_mm256_cvtps_pd(_mm256_extractf128_ps(reference, 1)));
				sumLow = _mm256_fmadd_pd(low, low, sumLow);
				sumHigh = _mm256_fmadd_pd(high, high, sumHigh);
			}
			squaredRankDifference = _mm256_reduceadd_pd(sumLow) +
				_mm256_reduceadd_pd(sumHigh);
		}
		for (size_t i = static_cast<size_t>(n8); i < count; ++i)
		{
			const double difference = static_cast<double>(predictorRank[i]) -
				static_cast<double>(referenceRank[i]);
			squaredRankDifference += difference * difference;
		}
		const double sampleCount = static_cast<double>(count);
		const double rankScale = (sampleCount * sampleCount * sampleCount -
			sampleCount) / 6.0;
		const double spearmanDenominator = std::sqrt(
			(rankScale - 2.0 * predictorTieScore) *
			(rankScale - 2.0 * referenceTieScore));
		if (spearmanDenominator > 0.0)
			result.spearman = (rankScale - squaredRankDifference -
				predictorTieScore - referenceTieScore) / spearmanDenominator;

		std::fill(fenwick.begin(), fenwick.end(), uint64_t{ 0 });
		const auto lowBit = [](const size_t value) { return value & (~value + 1); };
		const auto prefixSum = [&](size_t index)
		{
			uint64_t sum = 0;
			for (; index > 0; index -= lowBit(index)) sum += fenwick[index];
			return sum;
		};
		const auto add = [&](size_t index)
		{
			for (; index < fenwick.size(); index += lowBit(index)) ++fenwick[index];
		};
		uint64_t concordant = 0;
		uint64_t discordant = 0;
		uint64_t processed = 0;
		for (size_t begin = 0; begin < count;)
		{
			size_t end = begin + 1;
			while (end < count && predictor[order[end]] == predictor[order[begin]]) ++end;
			for (size_t i = begin; i < end; ++i)
			{
				const size_t rank = referenceOrdinalRank[order[i]];
				const uint64_t less = prefixSum(rank - 1);
				const uint64_t lessOrEqual = prefixSum(rank);
				concordant += less;
				discordant += processed - lessOrEqual;
			}
			for (size_t i = begin; i < end; ++i)
			{
				add(referenceOrdinalRank[order[i]]);
				++processed;
			}
			begin = end;
		}
		const double pairs = sampleCount * (sampleCount - 1.0) * 0.5;
		const double kendallDenominator = std::sqrt(
			(pairs - predictorTiedPairs) * (pairs - referenceTiedPairs));
		if (kendallDenominator > 0.0)
			result.kendall = (static_cast<double>(concordant) -
				static_cast<double>(discordant)) / kendallDenominator;

		double chatterjeeNumerator = 0.0;
		for (size_t i = 1; i < count; ++i)
		{
			const long long first = static_cast<long long>(
				referenceLessEqualRank[order[i - 1]]);
			const long long second = static_cast<long long>(
				referenceLessEqualRank[order[i]]);
			chatterjeeNumerator += static_cast<double>(std::llabs(second - first));
		}
		if (chatterjeeDenominator > 0.0)
			result.chatterjee = 1.0 - sampleCount * chatterjeeNumerator /
				chatterjeeDenominator;
		return result;
	}

	double CorrelationCoefficientSet::computePearson(
		const std::vector<double>& predictor) const
	{
		const size_t count = referenceCentered.size();
		if (predictor.size() != count || count < 2 ||
			!std::all_of(predictor.begin(), predictor.end(),
				[](const double value) { return std::isfinite(value); }))
			return std::numeric_limits<double>::quiet_NaN();
		return computePearsonUnchecked(predictor);
	}

	double CorrelationCoefficientSet::computePearsonUnchecked(
		const std::vector<double>& predictor) const noexcept
	{
		const size_t count = referenceCentered.size();
		const double predictorMean = std::accumulate(
			predictor.begin(), predictor.end(), 0.0) / static_cast<double>(count);
		double covariance = 0.0;
		double predictorVariance = 0.0;
		for (size_t i = 0; i < count; ++i)
		{
			const double centered = predictor[i] - predictorMean;
			covariance += centered * referenceCentered[i];
			predictorVariance += centered * centered;
		}
		const double denominator = std::sqrt(
			predictorVariance * referenceVariance);
		return denominator > 0.0 ? covariance / denominator :
			std::numeric_limits<double>::quiet_NaN();
	}
#pragma endregion

#pragma region Spearman

	SpearmanRankOrderCorrelationCoefficient::SpearmanRankOrderCorrelationCoefficient()
	{
		;
	}
#pragma region rankTransform
	template<typename T>
	double rankTransformUsingAverageTieScoreAVX2(
		vector<T>& src, vector<float>& dst, vector<size_t>& indices)
	{
		const size_t count = src.size();
		dst.resize(count);
		indices.resize(count);
		if constexpr (std::is_floating_point_v<T>)
			for (const T value : src)
				if (!std::isfinite(value))
					return std::numeric_limits<double>::quiet_NaN();
		std::iota(indices.begin(), indices.end(), size_t{ 0 });
		avx2_argsort(src.data(), indices.data(), count, false);

		double tie = 0.0;
		for (size_t begin = 0; begin < count;)
		{
			size_t end = begin + 1;
			while (end < count && src[indices[end]] == src[indices[begin]]) ++end;
			const float averageRank = 0.5f * static_cast<float>(begin + end - 1);
			for (size_t index = begin; index < end; ++index)
				dst[indices[index]] = averageRank;
			const double size = static_cast<double>(end - begin);
			tie += size * size * size - size;
			begin = end;
		}
		return tie / 12.0;
	}

	template<>
	double SpearmanRankOrderCorrelationCoefficient::rankTransformUsingAverageTieScore<int>(vector<int>& src, vector<float>& dst)
	{
		return rankTransformUsingAverageTieScoreAVX2(src, dst, indices);
	}

	template<>
	double SpearmanRankOrderCorrelationCoefficient::rankTransformUsingAverageTieScore<float>(vector<float>& src, vector<float>& dst)
	{
		return rankTransformUsingAverageTieScoreAVX2(src, dst, indices);
	}

	template<>
	double SpearmanRankOrderCorrelationCoefficient::rankTransformUsingAverageTieScore<double>(vector<double>& src, vector<float>& dst)
	{
		return rankTransformUsingAverageTieScoreAVX2(src, dst, indices);
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
			const float* s = &Rsrc[0];
			const float* r = &Rref[0];
			if (accumulation == Accumulation::Float32)
			{
				constexpr int chunkSize = 256;
				int i = 0;
				for (; i + 31 < N;)
				{
					const int end = std::min(N, i + chunkSize);
					__m256 sum0 = _mm256_setzero_ps();
					__m256 sum1 = _mm256_setzero_ps();
					__m256 sum2 = _mm256_setzero_ps();
					__m256 sum3 = _mm256_setzero_ps();
					for (; i + 31 < end; i += 32)
					{
						const __m256 sub0 = _mm256_sub_ps(
							_mm256_loadu_ps(s + i), _mm256_loadu_ps(r + i));
						const __m256 sub1 = _mm256_sub_ps(
							_mm256_loadu_ps(s + i + 8), _mm256_loadu_ps(r + i + 8));
						const __m256 sub2 = _mm256_sub_ps(
							_mm256_loadu_ps(s + i + 16), _mm256_loadu_ps(r + i + 16));
						const __m256 sub3 = _mm256_sub_ps(
							_mm256_loadu_ps(s + i + 24), _mm256_loadu_ps(r + i + 24));
						sum0 = _mm256_fmadd_ps(sub0, sub0, sum0);
						sum1 = _mm256_fmadd_ps(sub1, sub1, sum1);
						sum2 = _mm256_fmadd_ps(sub2, sub2, sum2);
						sum3 = _mm256_fmadd_ps(sub3, sub3, sum3);
					}
					const __m256 sum = _mm256_add_ps(
						_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
					ret += static_cast<double>(_mm256_reduceadd_ps(sum));
				}
				__m256 remainder = _mm256_setzero_ps();
				for (; i < N; i += 8)
				{
					const __m256 sub = _mm256_sub_ps(
						_mm256_loadu_ps(s + i), _mm256_loadu_ps(r + i));
					remainder = _mm256_fmadd_ps(sub, sub, remainder);
				}
				ret += static_cast<double>(_mm256_reduceadd_ps(remainder));
			}
			else
			{
				__m256d sumLow = _mm256_setzero_pd();
				__m256d sumHigh = _mm256_setzero_pd();
				for (int i = 0; i < N; i += 8)
				{
					const __m256 source = _mm256_loadu_ps(s + i);
					const __m256 reference = _mm256_loadu_ps(r + i);
					const __m256d subLow = _mm256_sub_pd(
						_mm256_cvtps_pd(_mm256_castps256_ps128(source)),
						_mm256_cvtps_pd(_mm256_castps256_ps128(reference)));
					const __m256d subHigh = _mm256_sub_pd(
						_mm256_cvtps_pd(_mm256_extractf128_ps(source, 1)),
						_mm256_cvtps_pd(_mm256_extractf128_ps(reference, 1)));
					sumLow = _mm256_fmadd_pd(subLow, subLow, sumLow);
					sumHigh = _mm256_fmadd_pd(subHigh, subHigh, sumHigh);
				}
				ret = _mm256_reduceadd_pd(sumLow) +
					_mm256_reduceadd_pd(sumHigh);
			}
			for (int i = N; i < n; i++)
			{
				if (accumulation == Accumulation::Float32)
				{
					const float sub = Rsrc[i] - Rref[i];
					ret += sub * sub;
				}
				else
				{
					const double sub = static_cast<double>(Rsrc[i]) -
						static_cast<double>(Rref[i]);
					ret += sub * sub;
				}
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
