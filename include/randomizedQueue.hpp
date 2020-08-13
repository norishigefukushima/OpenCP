#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT RandomizedQueue
	{
		cv::RNG rng;
	public:
		RandomizedQueue(cv::RNG& rng);
		RandomizedQueue(int64 state);

		std::deque<cv::Point> dq1;
		std::deque<cv::Point> dq2;

		bool empty();
		int size();
		void push(cv::Point pt);
		cv::Point pop();
	};
}
