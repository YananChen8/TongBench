#pragma once
#include <exception>
#include <grpcpp/grpcpp.h>
#include <stdexcept>

namespace tongos
{
	class RpcException : public std::exception
	{
	public:
		template <typename T>
		RpcException(T msg)
			: code(grpc::StatusCode::INTERNAL), msg(std::forward<T>(msg))
		{
		}

		template <typename T>
		RpcException(grpc::StatusCode code, T msg)
			: code(code), msg(std::forward<T>(msg))
		{
		}

		const ResponseStatus status() const { return {code, msg}; }

		const char* what() const noexcept { return msg.c_str(); }

	private:
		grpc::StatusCode code;
		std::string msg;
	};
} // namespace tongos
