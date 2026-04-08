#include "util/logger.h"
#include "util/channel.h"
#include "util/thread.h"
#include "util/time.h"
#include <fstream>
#include <thread>

namespace tongos
{
	int g_log_level = 0;
	std::ofstream g_log_file("../tongtest-grpc.log", std::ios_base::app);
	Channel<std::string> g_log_channel;

	void setLogLevel(int log_level)
	{
		g_log_level = log_level;

		// 重新打开一个日志文件
		g_log_file.close();
		std::ostringstream log_file_name_oss;
		log_file_name_oss << "../tongtest-grpc-";
		PUT_TIME(log_file_name_oss, "%Y%m%d-%H%M%S");
		log_file_name_oss << ".log";
		g_log_file = std::ofstream(log_file_name_oss.str(), std::ios_base::app);
	}

	void closeLogFile()
	{
		// g_log_file.close();
	}

	void asyncWriteLog()
	{
		while (true)
		{
			g_log_file << g_log_channel.receive().value();
			g_log_file.flush();
		}
	}

	std::thread launchLogThread()
	{
		std::thread log_thread(asyncWriteLog);
		log_thread.detach();
		return log_thread;
	}

	Logger::Logger(const char* filepath, int line, int level) : level_(level)
	{
		static std::thread log_thread = launchLogThread();
		// log_thread.detach();
		int64_t us = PUT_TIME(oss, "%Y-%m-%d %H:%M:%S.");
		oss << std::setfill('0')
			<< std::setw(6) << us << ' ' << getThreadId() << " ["
			<< basename(filepath) << ":" << line << "] ";
	}

	Logger::~Logger()
	{
		if (level_ < g_log_level)
		{
			return;
		}
		oss << std::endl;
		// std::cout << oss.str();
		g_log_channel.send(oss.str());
	}
} // namespace tongos
