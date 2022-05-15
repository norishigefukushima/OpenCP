#include "xmeans.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/hal.hpp>

namespace cp
{
	// improved xmeans (under debug)
	double log_likelihood(const cv::Mat& data, const cv::Vec3f& mu);
	double normalCDF(double value);
	double calcBIC(const cv::Mat& data, const cv::Vec3f& mu);
	double calcBIC2(const cv::Mat& data1, const cv::Mat& data2, const cv::Vec3f& mu1, const cv::Vec3f& mu2);
	cv::Mat cov(const cv::Mat& data, const cv::Vec3f& mu);

	// xmeans original
	double _log_likelihood(const cv::Mat& data, const cv::Vec3f& mu, int num_points, int num_center, double variance);
	double _cluster_variance(const cv::Mat& data, const cv::Vec3f& mu, int num_points, int num_center);
	double _calcBIC(const cv::Mat& data, const cv::Vec3f& mu);
	double _calcBIC2(const cv::Mat& data1, const cv::Mat& data2, const cv::Vec3f& mu1, const cv::Vec3f& mu2);

	void recursive_split(const cv::Mat& cluster, const cv::Vec3f& mu, cv::TermCriteria criteria, int attempts, int flags, cv::Mat& centers)
	{
		if (cluster.rows < 3)
		{
			centers.push_back(mu);
			return;
		}

		// devide
		cv::Mat lab, c;
		cv::kmeans(cluster, 2, lab, criteria, attempts, flags, c);
		//cp::kmeans(cluster, 2, lab, criteria, attempts, flags, c);

		cv::Mat cluster1, cluster2;

		for (int i = 0; i < cluster.rows; i++)
		{
			const cv::Vec3f* data_ptr = cluster.ptr<cv::Vec3f>(i);
			switch (lab.ptr<int>(i)[0])
			{
			case 0:
				cluster1.push_back(data_ptr[0]);
				break;
			case 1:
				cluster2.push_back(data_ptr[0]);
				break;
			default:
				break;
			}
		}

		cv::Vec3f mu1 = c.ptr<cv::Vec3f>(0)[0];
		cv::Vec3f mu2 = c.ptr<cv::Vec3f>(1)[0];

		double bic, bic2;
		//bic = calcBIC(cluster, mu);
		//std::cout << "BIC :" << bic << std::endl;
		//bic2 = calcBIC2(cluster1, cluster2, mu1, mu2);
		//std::cout << "BIC2:" << bic2 << std::endl;

		bic = _calcBIC(cluster, mu);
		bic2 = _calcBIC2(cluster1, cluster2, mu1, mu2);
		//std::cout << "BIC :" << bic << std::endl;
		//std::cout << "BIC2:" << bic2 << std::endl;

		if (bic2 > bic)
		{
			recursive_split(cluster1, mu1, criteria, attempts, flags, centers);
			recursive_split(cluster2, mu2, criteria, attempts, flags, centers);
		}
		else
		{
			//std::cout << "end" << std::endl;
			//std::cout << mu << std::endl;
			centers.push_back(mu);
		}
		return;

	}

	double log_likelihood(const cv::Mat& data, const cv::Vec3f& mu)
	{
		cv::Mat covar = cov(data, mu);
		double det_covar = cv::determinant(covar);
		double pi = (2 * CV_PI) * (2 * CV_PI) * (2 * CV_PI);
		double coef = 1.0 / sqrt(pi * det_covar);
		cv::Mat inv_covar = covar.inv();
		cv::Mat mu_(cv::Size(3, 1), CV_32F);
		mu_.ptr<cv::Vec3f>(0)[0] = mu;
		cv::Mat left(cv::Size(3, 1), CV_32F);
		cv::Mat right;
		//std::cout << mu_ << std::endl;
		//std::cout << covar << std::endl;
		//std::cout << "coef:" << coef << std::endl;

		// •ªŽU‹¤•ªŽUs—ñ‚Ìs—ñŽ®‚ª‚O‚Ì‚Æ‚«‚ªƒoƒO‚é?
		//std::cout << "det_covar:" << det_covar << std::endl;

		//if (det_covar < 0)
		//{
		//	std::cout << "det_covar is negative" << std::endl;
		//	//std::cout  << "det_covar "<< det_covar << std::endl;
		//	//return 0;
		//}

		cv::Mat value;
		double sum = 0;

		for (int i = 0; i < data.rows; i++)
		{
			left.ptr<cv::Vec3f>(0)[0] = data.ptr<cv::Vec3f>(i)[0];

			left = left - mu_;
			right = left.t();
			value = -0.5 * (left * (inv_covar * right));


			//if (coef * exp(value.ptr<float>(0)[0]) < 0)
			//{
			//// log‚Ì’†‚Í0ˆÈã
			//	std::cout << " log’†g:" <<coef*exp(value.ptr<float>(0)[0]) << std::endl;
			//}
			//std::cout << " log’†g:" <<coef*exp(value.ptr<float>(0)[0]) << std::endl;
			//if (value.ptr<float>(0)[0] < -100)
			//{
			//	return DBL_MIN;
			//}
			//else
			//{
			//	sum += std::log(coef * std::exp(value.ptr<float>(0)[0]));
			//}
			sum += std::log(coef * std::exp(value.ptr<float>(0)[0]));

			//std::cout << "value:" << value << std::endl;
			if (isinf(sum))
			{
				// log‚Ì’†g‚ª0‚É‚È‚Á‚Ä-inf‚É‚È‚Á‚Ä‚é
				std::cout << "INF IS OCCURED i:" << i << std::endl;
				std::cout << "value:" << value << std::endl;
				std::cout << "sum:" << sum << std::endl;
				std::cout << coef * std::exp(value.ptr<float>(0)[0]) << std::endl;
				return DBL_MIN;
			}
			//std::cout << "sum:" << sum << std::endl;
			//std::cout << left << std::endl;
			//std::cout << left*inv_covar*right << std::endl;
			//std::cout << right << std::endl;

		}
		//std::cout << "sum:" << sum << std::endl;

		return sum;
	}

	double normalCDF(double value)
	{
		return 0.5 * erfc(-value * (1 / sqrt(2)));
	}

	double calcBIC(const cv::Mat& data, const cv::Vec3f& mu)
	{
		double log_l = log_likelihood(data, mu);
		int d = data.channels();
		int q = d * (d + 3) / 2;
		//int q = 2 * d; /* ‹¤•ªŽU–³Ž‹@*/
		//std::cout << -2 * log_l * q * std::log(data.rows) << std::endl;
		//std::cout << "log_l:" << -2 * log_l << std::endl;
		if (log_l == DBL_MIN) return DBL_MAX;
		return -2 * log_l + q * std::log(data.rows);
	}

	double calcBIC2(const cv::Mat& data1, const cv::Mat& data2, const cv::Vec3f& mu1, const cv::Vec3f& mu2)
	{
		cv::Mat covar1, covar2;
		covar1 = cov(data1, mu1);
		covar2 = cov(data2, mu2);
		double det_covar1 = cv::determinant(covar1);
		double det_covar2 = cv::determinant(covar2);
		double beta;
		if (det_covar1 == 0 && det_covar2 == 0)
		{
			beta = 0;
		}
		else
		{
			beta = cv::norm(mu1, mu2) / sqrt(det_covar1 + det_covar2);
			//std::cout << "beta:"  << beta << std::endl;
		}
		double alpha = 0.5 / normalCDF(beta);
		int d = data1.channels();
		int q = d * (d + 3);
		//int q = 4 * d;@@/* ‹¤•ªŽU–³Ž‹@*/
		int cluster_size = data1.rows + data2.rows;
		double log_l1, log_l2;
		log_l1 = log_likelihood(data1, mu1);
		log_l2 = log_likelihood(data2, mu2);
		if (log_l1 == DBL_MIN || log_l2 == DBL_MIN) return DBL_MAX;
		//std::cout << "alpha:"  << alpha << std::endl;
		//std::cout << -2 * (cluster_size * std::log(alpha) + log_l1 + log_l2) + q * std::log(cluster_size) << std::endl;
		return -2 * (cluster_size * std::log(alpha) + log_l1 + log_l2) + q * std::log(cluster_size);
	}

	cv::Mat cov(const cv::Mat& data, const cv::Vec3f& mu)
	{
		cv::Mat reshaped_data = data.reshape(1, data.size().area());
		cv::Mat cov;
		cv::Mat mu_(cv::Size(3, 1), CV_32F);
		mu_.ptr<cv::Vec3f>(0)[0] = mu;
		cv::calcCovarMatrix(reshaped_data, cov, mu_, cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_USE_AVG, CV_32F);
		//// ‹¤•ªŽU–³Ž‹
		//cv::Mat m = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
		//cov = cov.mul(m);
		//std::cout << cov << std::endl;
		//cv::calcCovarMatrix(reshaped_data, cov, mu_, cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_USE_AVG | cv::COVAR_SCALE, CV_32F);
		//cov = cov.t();
		return cov;
	}

	double _log_likelihood(const cv::Mat& data, const cv::Vec3f& mu, int num_points, int num_center, double variance)
	{
		double ll = 0;
		int Rn = data.rows;
		int num_dims = data.channels();
		double t1 = Rn * log(Rn);
		double t2 = Rn * log(num_points);
		//std::cout << "variance:" << variance << std::endl;
		if (variance == 0 || isinf(variance))
		{
			return DBL_MAX;
		}
		//else
		//{

		//}
		double t3 = Rn * 0.5 * log(2 * CV_PI);
		double t4 = Rn * num_dims * 0.5 * log(variance);
		double t5 = (Rn - num_center) * 0.5;
		//double t4 = num_dims * (Rn - 1.0) / 2.0;
		ll += t1 - t2 - t3 - t4 - t5;
		//std::cout << "ll:" << ll << std::endl;
		//let L = -Rn * 0.5 * Math.log(2 * Math.PI) - Rn * dimension * 0.5 * Math.log(sigmaSquared) - (Rn - K) * 0.5 + Rn * Math.log(Rn) - Rn * Math.log(R_totalNumberOfPoints);
		return ll;
	}

	double _cluster_variance(const cv::Mat& data, const cv::Vec3f& mu, int num_points, int num_center)
	{
		double sum = 0;
		int num_dims = data.channels();
		int denom = (num_points - num_center);
		for (int i = 0; i < data.rows; i++)
		{
			const cv::Vec3f* data_ptr = data.ptr<cv::Vec3f>(i);
			sum += cv::norm(data_ptr[0], mu, cv::NORM_L2SQR);
			//std::cout << "sum:" << sum << std::endl;
			//std::cout << "data:" << data_ptr[0] << std::endl;
		}
		//std::cout << "sum:" << sum << std::endl;
		//std::cout << "denom:" << denom << std::endl;

		return sum / denom;
	}

	double _calcBIC(const cv::Mat& data, const cv::Vec3f& mu)
	{
		int num_points = data.rows;
		int num_dims = data.channels();
		int num_center = 1;
		double variance = _cluster_variance(data, mu, num_points, num_center);
		double log_likelihood = _log_likelihood(data, mu, num_points, num_center, variance);
		if (log_likelihood == DBL_MAX) return DBL_MAX;
		int num_params = (num_dims + 1);
		return log_likelihood - (num_params * 0.5) * log(num_points);
	}

	double _calcBIC2(const cv::Mat& data1, const cv::Mat& data2, const cv::Vec3f& mu1, const cv::Vec3f& mu2)
	{
		int num_points = data1.rows + data2.rows;
		int num_dims = data1.channels();
		int num_center = 2;
		double variance = _cluster_variance(data1, mu1, num_points, num_center) + _cluster_variance(data1, mu1, num_points, num_center);

		double log_likelihood1 = _log_likelihood(data1, mu1, num_points, num_center, variance);
		double log_likelihood2 = _log_likelihood(data2, mu2, num_points, num_center, variance);
		if (log_likelihood1 == DBL_MAX || log_likelihood2 == DBL_MAX) return DBL_MAX;
		int num_params = num_center * (num_dims + 1);
		return log_likelihood1 + log_likelihood2 - num_params * log(num_points);
	}

	int xmeans(cv::InputArray src, cv::OutputArray destLabels, const cv::TermCriteria criteria, const int attempts, const int flags, cv::OutputArray destCenters)
	{
		const cv::Mat data = src.getMat();
		cv::Mat labels = destLabels.getMat();
		cv::Mat centers = destCenters.getMat();

		cv::Mat lab, c;

		//cv::setRNGSeed(0);//to fix seed
		// defiding 2
		//cp::kmeans(data, 2, lab, criteria, attempts, flags, c);//sometimes stopped
		cv::kmeans(data, 2, lab, criteria, attempts, flags, c);
		//std::cout << lab << std::endl;
		cv::Mat cluster1, cluster2;
		//	for (int i = 0; i < data.rows; i++)
		//	{
		//		const cv::Vec3f* data_ptr = data.ptr<cv::Vec3f>(i);
		//		//std::cout << data_ptr[0]<< std::endl;
		//			cluster1.push_back(data_ptr[0]);
		//			//std::cout << cv::Mat3f(data_ptr[0][0], data_ptr[0][1], data_ptr[0][2]) << std::endl;
		//
		//		//std::cout << data_ptr[0][0] << std::endl;
		//		//std::cout << data_ptr[0][1] << std::endl;
		//		//std::cout << data_ptr[0][2] << std::endl;
		//		//std::cout << lab.ptr<int>(i)[0] << std::endl;
		//
		//}
		for (int i = 0; i < data.rows; i++)
		{
			const cv::Vec3f* data_ptr = data.ptr<cv::Vec3f>(i);
			switch (lab.ptr<int>(i)[0])
			{
			case 0:
				cluster1.push_back(data_ptr[0]);
				break;
			case 1:
				cluster2.push_back(data_ptr[0]);
				break;
			default:
				break;
			}
		}
		//std::cout << cluster1 << std::endl;
		//std::cout << cluster2 << std::endl;
		//std::cout << c.ptr<cv::Vec3f>(0)[0] << std::endl;
		//cv::Mat mu1 = c.ptr<cv::Vec3f>(0);
		//cluster1 = cluster1.reshape(1, cluster1.size().area());
		//cv::Mat covar = cov(cluster1, c.ptr<cv::Vec3f>(0)[0]);
		//cv::Mat covar;
		//cv::Mat mu(cv::Size(3,1), CV_32F);
		//mu.ptr<cv::Vec3f>(0)[0] = c.ptr<cv::Vec3f>(0)[0];
		//std::cout << c.ptr<cv::Vec3f>(0)[0] << std::endl;
		//std::cout << mu << std::endl;

		//std::cout << cluster1.channels() << std::endl;
		//cluster1 = cluster1.reshape(1, cluster1.size().area());
		//std::cout << cluster1.channels() << std::endl;

		//cv::calcCovarMatrix(cluster1, covar, mu, cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_USE_AVG , CV_32F);

		//std::cout << covar << std::endl;

		//log_likelihood(cluster1, c.ptr<cv::Vec3f>(0)[0]);
		//calcBIC(cluster1, c.ptr<cv::Vec3f>(0)[0]);
		//calcBIC(cluster2, c.ptr<cv::Vec3f>(1)[0]);
		//calcBIC2(cluster1,cluster2, c.ptr<cv::Vec3f>(0)[0], c.ptr<cv::Vec3f>(1)[0]);


		//double norm = cv::norm(c.ptr<cv::Vec3f>(0)[0], c.ptr<cv::Vec3f>(0)[1]);

		//std::cout << c.ptr<cv::Vec3f>(0)[0] << std::endl;
		//std::cout << c.ptr<cv::Vec3f>(0)[1] << std::endl;
		//std::cout << norm << std::endl;

		recursive_split(cluster1, c.ptr<cv::Vec3f>(0)[0], criteria, attempts, flags, centers);
		recursive_split(cluster2, c.ptr<cv::Vec3f>(1)[0], criteria, attempts, flags, centers);

		//std::cout << "num of clusters: " << centers.rows << std::endl;

		return centers.rows;

	}
}