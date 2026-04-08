#pragma once
#include "Containers/StringConv.h"
#include <chrono>
#include <ctime>
// #include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstring>

namespace tongos
{
	constexpr int DEBUG = 0;
	constexpr int INFO = 1;
	constexpr int WARN = 2;
	// constexpr int ERROR = 3;
	constexpr int FATAL = 4;

	void TONGOSGRPC_API setLogLevel(int log_level);
	void TONGOSGRPC_API closeLogFile();
	// class GLOG_EXPORT LogStreamBuf : public std::streambuf {
	// public:
	//   LogStreamBuf(char *buf, int len) { setp(buf, buf + len - 2); }

	//   // This effectively ignores overflow.
	//   int_type overflow(int_type ch) { return ch; }
	// };


	static const char* basename(const char* filepath)
	{
		const char* base = strrchr(filepath, '/');
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
		if (!base)
			base = strrchr(filepath, '\\');
#endif
		return base ? (base + 1) : filepath;
	}

	class TONGOSGRPC_API Logger
	{
	public:
		Logger(const char* filepath, int line, int level);
		~Logger();

		template <typename T>
		friend std::ostream& operator<<(Logger&& logger, T&& t)
		{
			logger.oss << t;
			return logger.oss;
		}

	private:
		std::ostringstream oss;
		int level_;
	};

	class BlankLogger
	{
	public:
		BlankLogger()
		{
		}

		template <typename T>
		friend const BlankLogger& operator<<(const BlankLogger& logger, T&& t)
		{
			return logger;
		}
	};

	// #define tonglog(INFO)                                                              \
	//   BlankLogger {}
#define tonglog(level)                                                              \
  Logger { __FILE__, __LINE__, level}
	// #define tonglog(level) LOG(level)
} // namespace tongos
