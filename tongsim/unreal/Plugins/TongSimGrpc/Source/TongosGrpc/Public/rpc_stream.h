#pragma once

#include "rpc_common.h"
#include "rpc_event.h"
#include "rpc_exception.h"
#include "rpc_type.h"
#include "rpc_write_queue.h"
// #include "util/logger.h"

#include <grpcpp/generic/async_generic_service.h>
#include <grpcpp/grpcpp.h>
// #include <grpcpp/impl/proto_utils.h>

#include <functional>
#include <optional>
#include <string_view>
#include <sstream>

namespace tongos
{
	enum class RpcState : uint8_t
	{
		NORMAL = 0,
		// ERROR = 1,
		CANCELLED = 2,
		FINISHED = 3,
	};

	class RpcReactorBase;

	class RpcStream
	{
	public:
		using Ptr = std::shared_ptr<RpcStream>;

		enum OpTag
		{
			CALL = 0,
			READ = 1,
			WRITE = 2,
			DONE = 3,
			FINISH = 7,
		};

		RpcStream(grpc::AsyncGenericService* generic_service,
		          grpc::ServerCompletionQueue* cq, RpcTypeQueryer& rpc_type_queryer)
			: generic_service(generic_service), cq(cq),
			  rpc_type_queryer(rpc_type_queryer), generic_server_ctx(),
			  generic_stream(&generic_server_ctx), rpc_type(RpcType::UNARY),
			  writing(false), rpc_state_(RpcState::NORMAL), rpc_reactor_(nullptr)
		{
			// TODO:
			// tonglog(DEBUG) << (void*)this << ": RpcStream constructed";
			generic_server_ctx.AsyncNotifyWhenDone(encodeTag(OpTag::DONE));
			generic_service->RequestCall(&generic_server_ctx, &generic_stream, cq, cq,
			                             encodeTag(OpTag::CALL));
		}


		~RpcStream()
		{
			// TODO:
			// tonglog(DEBUG) << (void*)this << ": RpcStream destructed";
		}

		void bindRpcReactor(RpcReactorBase* rpc_reactor)
		{
			this->rpc_reactor_ = rpc_reactor;
		}

		RpcReactorBase* getRpcReactor() { return this->rpc_reactor_; }

		static std::optional<RpcEvent> handle(void* tag, bool ok,
		                                      std::function<void()>& increment_hook,
		                                      std::function<void()>& decrement_hook)
		{
			RpcStream* rpc_stream;
			uint8_t op_tag;
			rpc_stream = decodeTag(tag, op_tag);
			// TODO:
			// tonglog(DEBUG) << (void*)rpc_stream << ", op:" << (uint32_t)op_tag
			// 	<< ", ok:" << std::boolalpha << ok;
			return rpc_stream->parse(op_tag, ok, increment_hook, decrement_hook);
		}

		void readToBuffer()
		{
			// TODO:
			// tonglog(DEBUG) << (void*)this << ": read";
			if (rpc_state_ != RpcState::NORMAL)
			{
				// tonglog(INFO) << (void*)this << "rpc is cancelled: "
					// << (uint8_t)rpc_state_.load();
				return;
			}
			generic_stream.Read(&read_buffer, encodeTag(OpTag::READ));
		}

		grpc::ByteBuffer& readBuffer()
		{
			return read_buffer;
		}

		void write(grpc::ByteBuffer write_buffer)
		{
			std::scoped_lock guard(write_mu);
			// tonglog(DEBUG) << (void*)this << ": write";
			if (rpc_state_ != RpcState::NORMAL)
			{
				std::ostringstream oss;
				oss << (void*)this
					<< " rpc is cancelled: " << (uint16_t)rpc_state_.load();
				throw RpcException(grpc::StatusCode::CANCELLED, oss.str());
				return;
			}

			if (!writing)
			{
				generic_stream.Write(write_buffer, encodeTag(OpTag::WRITE));
				writing = true;
			}
			else
			{
				write_queue.emplace(std::move(write_buffer));
			}
		}

		void tryCancel() { generic_server_ctx.TryCancel(); }

		// 上层需要保证finish只被调用一次
		void finish(const ResponseStatus& status)
		{
			// tonglog(DEBUG) << (void*)this << ": finish code: " << status.error_code()
				// << ", msg:" << status.error_message();
			std::scoped_lock guard(write_mu);
			if (writing)
			{
				status_ = status;
			}
			else
			{
				doFinish(status);
			}
		}

		const std::string& method() const { return generic_server_ctx.method(); }
		RpcState rpcState() const { return rpc_state_; }

	private:
		void* encodeTag(uint8_t op_tag) { return (uint8_t*)this + op_tag; }

		static RpcStream* decodeTag(void* tag, uint8_t& op_tag)
		{
			op_tag = (uint64_t)tag & 0x07;
			RpcStream* rpc_stream = (RpcStream*)((uint64_t)tag ^ op_tag);

			return rpc_stream;
		}

		void doFinish(const ResponseStatus& status)
		{
			rpc_state_ = RpcState::FINISHED;
			generic_stream.Finish(status.to_grpc_status(), encodeTag(OpTag::FINISH));
		}

		RpcEvent genMessage(RpcEventType rpc_event_type)
		{
			return {this, rpc_reactor_, rpc_event_type};
		}

		std::optional<RpcEvent> onDone()
		{
			if (generic_server_ctx.IsCancelled())
			{
				// tonglog(INFO) << (void*)this << " client or server canclled";
				rpc_state_ = RpcState::CANCELLED;
				return genMessage(RpcEventType::CANCEL);
			}
			return {};
		}

		RpcEvent onFinish()
		{
			RpcReactorBase* rpc_reactor = this->rpc_reactor_;
			// 生命周期结束
			delete this;
			return RpcEvent(nullptr, rpc_reactor, RpcEventType::FINISH);
		}

		void parseRequest()
		{
			new RpcStream(generic_service, cq, rpc_type_queryer);
			// tonglog(INFO) << (void*)this << ": method: " << generic_server_ctx.method();
			auto optional_value =
				rpc_type_queryer.queryRpcType(generic_server_ctx.method());
			if (optional_value)
			{
				rpc_type = optional_value.value();
			}
		}

		std::optional<RpcEvent> parseUnary(uint8_t op_tag, bool ok)
		{
			std::optional<RpcEvent> message;
			if (!ok && op_tag != OpTag::WRITE)
			{
				// rpc_state_  = RpcState::ERROR;
				return message;
			}
			switch (op_tag)
			{
			case OpTag::CALL:
				readToBuffer();
				break;
			case OpTag::READ:
				message = genMessage(RpcEventType::REQUEST);
				break;
			case OpTag::WRITE:
				nextWrite();
				break;
			default:
				throw RpcException("unknown op tag:" + std::to_string(op_tag));
			}
			return message;
		}

		void nextWrite()
		{
			std::scoped_lock guard(write_mu);
			if (!write_queue.empty())
			{
				if (rpc_state_ == RpcState::NORMAL)
				{
					generic_stream.Write(write_queue.front(), encodeTag(OpTag::WRITE));
					write_queue.pop();
					return;
				}
				// tonglog(INFO) << (void*)this << ": " << generic_server_ctx.method() << ", nextWrite: rpc_state unnormal";
			}

			writing = false;
			if (status_)
			{
				doFinish(status_.value());
			}
		}

		std::optional<RpcEvent> parseServerStreaming(uint8_t op_tag, bool ok)
		{
			std::optional<RpcEvent> message;
			if (!ok && op_tag != OpTag::WRITE)
			{
				// rpc_state_  = RpcState::ERROR;
				return message;
			}
			switch (op_tag)
			{
			case OpTag::CALL:
				readToBuffer();
				break;
			case OpTag::READ:
				message = genMessage(RpcEventType::REQUEST);
				break;
			case OpTag::WRITE:
				nextWrite();
				break;
			default:
				throw RpcException("unknown op tag:" + std::to_string(op_tag));
			}
			return message;
		}

		std::optional<RpcEvent> parseBidiStreaming(uint8_t op_tag, bool ok)
		{
			std::optional<RpcEvent> message;
			switch (op_tag)
			{
			case OpTag::CALL:
				if (ok)
				{
					message = genMessage(RpcEventType::CALL);
				}
				break;
			case OpTag::READ:
				if (ok)
				{
					message = genMessage(RpcEventType::REQUEST);
				}
				else
				{
					message = genMessage(RpcEventType::REQUEST_DONE);
				}
				break;
			case OpTag::WRITE:
				nextWrite();
				break;
			default:
				throw RpcException("unknown op tag:" + std::to_string(op_tag));
			}
			return message;
		}

		std::optional<RpcEvent> parse(uint8_t op_tag, bool ok,
		                              std::function<void()>& increment_hook,
		                              std::function<void()>& decrement_hook)
		{
			if (op_tag == CALL)
			{
				if (!ok)
				{
					decrement_hook();
					onFinish();
					return {};
				}
				increment_hook();
				parseRequest();
			}
			else if (op_tag == OpTag::DONE)
			{
				return onDone();
			}
			else if (op_tag == OpTag::FINISH)
			{
				decrement_hook();
				return onFinish();
			}

			switch (rpc_type)
			{
			case RpcType::UNARY:
				return parseUnary(op_tag, ok);
				break;
			case RpcType::SERVER_STREAMING:
				return parseServerStreaming(op_tag, ok);
				break;
			case RpcType::BIDI_STREAMING:
				return parseBidiStreaming(op_tag, ok);
				break;
			default:
				throw RpcException("unknown method type");
			}
		}

	private:
		grpc::AsyncGenericService* generic_service;
		grpc::ServerCompletionQueue* cq;
		RpcTypeQueryer& rpc_type_queryer;
		grpc::GenericServerContext generic_server_ctx;
		grpc::GenericServerAsyncReaderWriter generic_stream;

		RpcType rpc_type;
		grpc::ByteBuffer read_buffer;

		std::mutex write_mu;
		bool writing;
		RpcWriteQueue write_queue;
		std::atomic<RpcState> rpc_state_;
		std::optional<ResponseStatus> status_;
		std::atomic<RpcReactorBase*> rpc_reactor_;
	};
} // namespace tongos
