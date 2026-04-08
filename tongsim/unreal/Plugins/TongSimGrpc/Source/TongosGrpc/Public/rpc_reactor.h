#pragma once
// #pragma warning(push)
#pragma warning(disable:4250)
#include "rpc_responder.h"
#include "rpc_reactor_base.h"

namespace tongos
{
	template <typename Request, typename Response>
	class RpcReactorUnary : public detail::RpcReactorUnaryBase<Request>
	{
	public:
		static constexpr RpcType rpc_type = RpcType::UNARY;
		// 写入一条消息，然后关闭流
		void writeAndFinish(const Response& response)
		{
			this->rpc_stream_rw->writeAndFinish(response);
		}
	};

	template <typename Request, typename Response>
	class RpcReactorUnarySync : public RpcReactorUnary<Request, Response>
	{
	public:
		virtual ResponseStatus onRequestSync(Request& request, Response& response) = 0;

	private:
		void onRequest(Request& request) override final
		{
			Response response;
			ResponseStatus status = onRequestSync(request, response);
			if (status.ok())
			{
				this->rpc_stream_rw->writeAndFinish(response);
			}
			else
			{
				this->rpc_stream_rw->finish(status);
			}
		}
	};

	template <typename Request, typename Response>
	class RpcReactorUnarySyncHandler
		: public RpcReactorUnarySync<Request, Response>
	{
	public:
		using Handler = std::function<ResponseStatus(Request&, Response&)>;

		static RpcReactorUnarySyncHandler* create(Handler handler)
		{
			return new RpcReactorUnarySyncHandler(handler);
		}

		ResponseStatus onRequestSync(Request& request,
		                             Response& response) override final
		{
			return handler(request, response);
		}

	private:
		RpcReactorUnarySyncHandler(Handler handler) : handler(handler)
		{
		}

		Handler handler;
	};

	template <typename Request, typename Response>
	class RpcReactorServerStreaming : public detail::RpcReactorUnaryBase<Request>,
	                                  public detail::RpcReactorServerStreamingInterface<Response>
	{
	public:
		static constexpr RpcType rpc_type = RpcType::SERVER_STREAMING;
		void write(const Response& response) override { this->rpc_stream_rw->write(response); }
	};

	template <typename Request, typename Response>
	class RpcReactorBidiStreaming : public RpcReactorBase
	{
	public:
		static constexpr RpcType rpc_type = RpcType::BIDI_STREAMING;
		// 写入一条消息
		void write(const Response& response) { rpc_stream_rw->write(response); }

	protected:
		// 注意onXXX之类的回调函数以及reactor的析构函数里,绝对不能调用阻塞操作，比如thread.join()，否则消息的处理会被卡住
		// 连接建立后调用
		virtual void onCall() = 0;
		// 收到请求消息后调用，如果request是空的，意味着客户端已经发送完毕所有请求或者已经断开连接
		virtual void onRequest(std::optional<Request>& request) = 0;

	private:
		void nextRequest() { rpc_stream_rw->readToBuffer(); }

		void handleCall(RpcEvent& rpc_event) override
		{
			onCall();
			nextRequest();
		}

		void handleRequest(RpcEvent& rpc_event) override
		{
			std::optional<Request> optional_request;
			optional_request.emplace();
			rpc_stream_rw->deserialize(optional_request.value());


			// Grpc Message Debug by WuKunKun:
			UTSGrpcMessageDebugSubsystem* DebugSubsystem = UTSGrpcMessageDebugSubsystem::GetInstance();
			if (optional_request && DebugSubsystem)
			{
				// TODO: Add method to serialize?
				// rpc_event.rpc_stream()->method()
				DebugSubsystem->DebugRequest(&optional_request.value());
			}

			onRequest(optional_request);
			nextRequest();
		}

		void handleRequestDone(RpcEvent& rpc_event) override
		{
			std::optional<Request> optional_request;
			onRequest(optional_request);
		}
	};

	template <typename Request, typename Response>
	class RpcReactorServerStreamingHandler
		: public RpcReactorServerStreaming<Request, Response>
	{
	public:
		// using Reactor = RpcReactorServerStreaming<Request, Response>;
		using Handler = std::function<void(Request&, RpcServerStreamingResponder<Response>)>;

		static RpcReactorServerStreamingHandler* create(Handler handler)
		{
			return new RpcReactorServerStreamingHandler(handler);
		}

		void onRequest(Request& request) override final
		{
			// request_ = std::move(request);
			handler(request,
			        RpcServerStreamingResponder<Response>{
				        {
					        this->template sharedSelf<RpcReactorServerStreamingHandler>(),
					        dynamic_cast<detail::RpcReactorServerStreamingInterface<Response>*>(this)
				        }
			        }
			);
		}

		Request& getRequest()
		{
			return request_;
		}

	private:
		RpcReactorServerStreamingHandler(Handler handler) : handler(handler)
		{
		}

		Handler handler;
		Request request_;
	};
} // namespace tongos
// #pragma warning(pop)
