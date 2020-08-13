#include "randomizedQueue.hpp"

using namespace cv;
using namespace std;

namespace cp
{

	RandomizedQueue::RandomizedQueue(RNG& rng)
	{
		this->rng = rng;
	}

	RandomizedQueue::RandomizedQueue(int64 state)
	{
		rng.state = state;
	}
	// Random Queue
	bool RandomizedQueue::empty()
	{
		return dq1.empty();
	}

	int RandomizedQueue::size()
	{
		return (int)dq1.size();
	}

	void RandomizedQueue::push(Point pt)
	{
		dq1.push_front(pt);
	}

	cv::Point RandomizedQueue::pop()
	{
		int n = rng.uniform(0, (int)dq1.size());

		Point pt;
		while (n--)
		{
			dq2.push_front(dq1.front());
			dq1.pop_front();
		}

		pt = dq1.front();
		dq1.pop_front();

		while (!dq2.empty())
		{
			dq1.push_front(dq2.front());
			dq2.pop_front();
		}
		return pt;
	}
}
