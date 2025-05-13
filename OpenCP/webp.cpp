#include "webp.hpp"
#include "./webp/encode.h"
#include "./webp/decode.h"
#include "./webp/mux.h"
#pragma comment(lib,"./webp/libwebp.lib")
#pragma comment(lib,"./webp/libwebpdecoder.lib")
#pragma comment(lib,"./webp/libwebpmux.lib")
#pragma comment(lib,"./webp/libwebpdemux.lib")
#pragma comment(lib,"./webp/libsharpyuv.lib")

#include "debugcp.hpp"

using namespace cv;
using namespace std;

namespace cp
{
	int imwriteAnimationWebp(std::string name, vector<Mat>& src, const vector<int>& parameters)
	{
		float quality = 100.f;
		int loop_count = 0;//0: infinit
		int timems_per_frame = 33;
		int method = 4;
		int colorspace = 0;//0: YUV, 1: RGB, 2: YUV_SHARP
		for (int i = 0; i < parameters.size(); i += 2)
		{
			if (parameters[i] == IMWRITE_WEBP_QUALITY)
			{
				quality = (float)parameters[i + 1];
			}
			if (parameters[i] == IMWRITE_WEBP_METHOD)
			{
				method = parameters[i + 1];
			}
			if (parameters[i] == IMWRITE_WEBP_LOOPCOUNT)
			{
				loop_count = parameters[i + 1];
			}
			if (parameters[i] == IMWRITE_WEBP_TIMEMSPERFRAME)
			{
				timems_per_frame = parameters[i + 1];
			}
			if (parameters[i] == IMWRITE_WEBP_COLORSPACE)
			{
				colorspace = parameters[i + 1];
			}
		}
		const int width = src[0].cols;
		const int height = src[0].rows;

		WebPPicture pic;
		if (!WebPPictureInit(&pic))
		{
			cerr << "picture init" << endl;
			return -1;  // version error
		}
		pic.width = width;
		pic.height = height;

		// allocated picture of dimension width x height
		if (!WebPPictureAlloc(&pic))
		{
			cerr << "picture alloc" << endl;
			return -1;   // memory error
		}

		WebPAnimEncoderOptions enc_options = { {0} };
		WebPAnimEncoderOptionsInit(&enc_options);
		//enc_options.kmax = 1;
		//enc_options.kmin = 256/2+1;//kmin >= kmax / 2 + 1
		WebPAnimEncoder* enc = WebPAnimEncoderNew(width, height, &enc_options);

		WebPConfig config;
		WebPConfigInit(&config);
		config.quality = quality;//0.f-100.f
		config.method = method;//0(fast)-6(slower, better), default 4
		if (!WebPValidateConfig(&config))
		{
			cerr << "config error" << endl;
			return -1;   // config error
		}

		Mat temp;
		for (int i = 0; i < src.size(); i++)
		{
			cvtColor(src[i], temp, COLOR_BGR2RGBA);

			if (!WebPPictureImportRGBA(&pic, temp.data, (int)temp.step))
			{
				cerr << "error: WebPPictureImportRGBA" << endl;
			}

			if (colorspace == 0) WebPPictureARGBToYUVA(&pic, WebPEncCSP::WEBP_YUV420);
			if (colorspace == 1) WebPPictureSharpARGBToYUVA(&pic);

			WebPAnimEncoderAdd(enc, &pic, timems_per_frame * i, &config);
		}

		// add a last fake frame to signal the last duration
		WebPAnimEncoderAdd(enc, NULL, timems_per_frame * (int)src.size(), NULL);

		WebPData webp_data = { 0 };
		WebPDataInit(&webp_data);

		//write data
		WebPAnimEncoderAssemble(enc, &webp_data);

		//Mux assemble
		WebPMux* mux = WebPMuxCreate(&webp_data, 1);
		WebPMuxAnimParams anim_params;
		anim_params.loop_count = loop_count;
		WebPMuxSetAnimationParams(mux, &anim_params);
		WebPMuxAssemble(mux, &webp_data);

		FILE* fp = fopen(name.c_str(), "wb");
		fwrite(webp_data.bytes, sizeof(char), webp_data.size, fp);
		const int ret = (int)webp_data.size;
		fclose(fp);
		WebPPictureFree(&pic);
		WebPAnimEncoderDelete(enc);
		WebPDataClear(&webp_data);
		WebPMuxDelete(mux);
		return ret;
	}

	int imencodeWebP(const Mat& src, vector<uchar>& buff, const vector<int>& parameters)
	{
		const int width = src.cols;
		const int height = src.rows;
		uint8_t* rgb_data = src.data;

		// WebP setting
		WebPConfig config;
		WebPPicture picture;
		WebPMemoryWriter writer;
		WebPPreset preset = WebPPreset::WEBP_PRESET_DEFAULT;

		if (!WebPConfigPreset(&config, preset, 75) || !WebPPictureInit(&picture))
		{
			fprintf(stderr, "WebP init error\n");
			return 1;
		}

		int color = 0;
		for (int n = 0; n < parameters.size(); n += 2)
		{
			if (parameters[n] == IMWRITE_WEBP_QUALITY) config.quality = (float)parameters[n + 1];
			if (parameters[n] == IMWRITE_WEBP_METHOD) config.method = parameters[n + 1];
			if (parameters[n] == IMWRITE_WEBP_COLORSPACE) color = parameters[n + 1];

		//	std::cout << n<<": "<<parameters[n] << "," << parameters[n + 1] << endl;
		}

		picture.width = width;
		picture.height = height;
		if (color == 0 || color == 1)
		{
			picture.use_argb = 1;
			if (color == 1) config.use_sharp_yuv = 1;
			WebPPictureImportBGR(&picture, rgb_data, width * 3);
			//WebPPictureARGBToYUVA(&picture, WebPEncCSP::WEBP_YUV420);
		}
		else
		{
			picture.use_argb = 1;
			WebPPictureImportBGR(&picture, rgb_data, width * 3);
		}
		WebPMemoryWriterInit(&writer);
		picture.writer = WebPMemoryWrite;
		picture.custom_ptr = &writer;

		if (!WebPEncode(&config, &picture))
		{
			fprintf(stderr, "compression error: %d\n", WebPEncodingError(picture.error_code));
			WebPPictureFree(&picture);
			//free(rgb_data);
			return 1;
		}

		buff.resize(writer.size);
		for (int i = 0; i < writer.size; i++)
		{
			buff[i] = writer.mem[i];
		}

		WebPPictureFree(&picture);
		WebPMemoryWriterClear(&writer);

		return 0;

	}
}