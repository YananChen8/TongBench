#include "util/time.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#include "Misc/DateTime.h"
#else
#include <iomanip>
#endif

namespace tongos
{
#ifdef __unix__
int64_t putTime(std::ostream&oss, const char* format){
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm ltime;
    localtime_r(&t, &ltime);

    oss << std::put_time(&ltime, format);
    int64_t us = std::chrono::time_point_cast<std::chrono::microseconds>(now)
                  .time_since_epoch()
                  .count() %
              1000000;
    return us;
}
#else
	int64_t putTime(std::ostream& oss, const wchar_t* format)
	{
		FDateTime fnow = FDateTime::Now();
		fnow.GetTicks();
		oss << TCHAR_TO_UTF8(*fnow.ToString(format));
		int64_t us = fnow.GetTicks() / 10 % 1000000;
		return us;
	}
#endif
}
