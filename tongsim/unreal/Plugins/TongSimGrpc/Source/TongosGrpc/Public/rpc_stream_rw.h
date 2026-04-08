#pragma once

#include "rpc_stream.h"
#include <sstream>
#include <grpcpp/support/proto_buffer_reader.h>
#include <grpcpp/impl/codegen/proto_utils.h>

namespace tongos
{
	// TODO这里的锁通过重载->单独提取出来
	class RpcStreamRW
	{
	public:
		RpcStreamRW() : rpc_stream_(nullptr), finished(false)
		{
		}

		~RpcStreamRW()
		{
		}

		void bind(RpcStream* rpc_stream) { this->rpc_stream_ = rpc_stream; }

		// readToBuffer不会抛出异常
		void readToBuffer()
		{
			std::scoped_lock lock_guard(mu);
			if (finished)
			{
				return;
			}

			return rpc_stream_->readToBuffer();
		}

		// 如果流已经被客户端/服务器cancel，或者finish了，会抛出异常
		// 如果解析请求失败，会抛出异常
		template <typename PB_TYPE>
		void deserialize(PB_TYPE& request)
		{
			std::scoped_lock lock_guard(mu);
			ensureUnFinished();

			grpc::Status status =
				grpc::GenericDeserialize<grpc::ProtoBufferReader, PB_TYPE>(
					&rpc_stream_->readBuffer(), &request);
			if (!status.ok())
			{
				std::ostringstream oss;
				oss << "method " << rpc_stream_->method()
					<< " deserialize request message failed, code:" << status.error_code()
					<< ", msg:" << status.error_message();
				throw RpcException{grpc::StatusCode::INVALID_ARGUMENT, oss.str()};
			}
		}

		// 如果流已经被客户端/服务器cancel，或者finish了，会抛出异常
		// 如果序列化响应失败，会抛出异常
		template <typename PB_TYPE>
		void write(PB_TYPE&& pb_value)
		{
			std::scoped_lock lock_guard(mu);
			ensureUnFinished();

			bool own_buffer;
			grpc::ByteBuffer write_buffer;
			auto status = grpc::GenericSerialize<grpc::ProtoBufferWriter, PB_TYPE>(
				std::forward<PB_TYPE>(pb_value), &write_buffer, &own_buffer);
			if (!status.ok())
			{
				std::ostringstream oss;
				oss << (void*)this << " " << rpc_stream_->method()
					<< " message serialize failed, code:" << status.error_code()
					<< ", msg:" << status.error_message();
				throw RpcException{grpc::StatusCode::INTERNAL, oss.str()};
			}

			return rpc_stream_->write(std::move(write_buffer));
		}

		// tryCancel不会抛异常
		void tryCancel()
		{
			std::scoped_lock lock_guard(mu);
			if (finished)
			{
				return;
			}

			rpc_stream_->tryCancel();
		}

		// finish不会抛异常
		void finish(const ResponseStatus& status)
		{
			std::scoped_lock lock_guard(mu);
			if (finished)
			{
				return;
			}

			rpc_stream_->finish(status);
			finished = true;
		}

		// writeAndFinish不会抛异常
		template <typename PB_TYPE>
		void writeAndFinish(PB_TYPE&& pb_value)
		{
			ResponseStatus status;
			try
			{
				write(std::forward<PB_TYPE>(pb_value));
			}
			catch (RpcException& ex)
			{
				status = ex.status();
			}
			finish(status);
		}

	private:
		void ensureUnFinished()
		{
			if (finished)
			{
				std::ostringstream oss;
				oss << (void*)this << " already finished";
				throw RpcException(grpc::StatusCode::ABORTED, oss.str());
			}
		}

		std::mutex mu;
		RpcStream* rpc_stream_;
		bool finished;
	};
} // namespace tongos
