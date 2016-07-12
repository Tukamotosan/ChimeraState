#include "ChimeraSim.h"
#include <time.h>

/*!
@brief OpenCVのサンプルプロジェクト詳細は以下を参照
http://www.buildinsider.net/small/opencv/003
http://www.buildinsider.net/small/opencv/004
http://qiita.com/konta220/items/f23f50decbae1133d198
*/
int main(int argc, const char* argv[])
{
	ChimeraSim sim;
	sim.init(0.2, -0.7, 0.1, 0.66, 256, 256, 0.2);
	int n = 10000;

	for(int i=0; i<n; i++){
		if ((i % 500) == 0) {
			sim.save_w_real("data\\w_real_" + std::to_string(sim.t) + ".jpg");
			//sim.save_w_imag("data\\w_imag_" + std::to_string(sim.t) + ".jpg");
			sim.save_txt("data\\out_" + std::to_string(sim.t) + ".txt");
		}
		clock_t start = clock();
		sim.exec();
		clock_t end = clock();
		printf("%d/%d duration is %.5f[sec]\n",( i + 1), n, (double)(end-start)/CLOCKS_PER_SEC);

	}
	return 0;
}