#pragma once
#include <ostream>

namespace tongos
{
#ifdef __unix__
int64_t putTime(std::ostream&oss, const char* format);
#define PUT_TIME(oss, format) putTime(oss, format);
#else
	int64_t putTime(std::ostream& oss, const wchar_t* format);
#define PUT_TIME(oss, format) putTime(oss, L##format);
#endif
}
