#pragma once
#include "rpc_stream_rw.h"
#include "CoreMinimal.h"
#include "Debug/TSGrpcMessageDebugSubsystem.h"

namespace tongos
{
	namespace detail
	{
		template <typename Request>
		class RpcReactorUnaryBase;
		template <typename Request, typename Response>
		class RpcReactorUnarySync;

		class RpcReactorFinisher
		{
		public:
			virtual void finish(const ResponseStatus&) = 0;
		};
	} // namespace detail


	class RpcReactorBase : virtual public detail::RpcReactorFinisher
	{
	public:
		RpcReactorBase()
		{
			rpc_stream_rw = MakeShareable(new RpcStreamRW);
		}

		using unique_ptr = std::unique_ptr<RpcReactorBase>;

		virtual ~RpcReactorBase()
		{
			// tonglog(DEBUG) << (void*)this << " reactor destructed";
		};

		// 尝试取消当前流的处理
		void tryCancel(ResponseStatus status) { rpc_stream_rw->tryCancel(); }

		// reactor需要保证在流处理结束后（不管处理是否出错）都要调用finish，否则客户端连接将不会中断
		// finish后不能再进行write和read操作
		void finish(const ResponseStatus& status) override { rpc_stream_rw->finish(status); }

		// reactor启动新线程的时候一定要调用sharedSelf，否则this指针可能会提前失效，导致段错误/死锁等各种奇怪的问题
		template <typename Reactor>
		std::shared_ptr<Reactor> sharedSelf()
		{
			// todo UE好像不能使用dynamic_cast？
			return {shared_self_, (Reactor*)(shared_self_.get())};
			// return {shared_self_, dynamic_cast<Reactor*>(shared_self_.get())};
		}

		std::shared_ptr<RpcReactorBase> sharedSelf() { return shared_self_; }

	protected:
		// client断开连接时或者server端调用tryCancel后触发
		// 这个回调能及时感知到错误，所以可以用来及时取消server正在进行的计算
		virtual void onCancel()
		{
		}

	private:
		friend class RpcRouter;
		template <typename Request>
		friend class detail::RpcReactorUnaryBase;
		template <typename Request, typename Response>
		friend class RpcReactorUnary;
		template <typename Request, typename Response>
		friend class RpcReactorUnarySync;
		template <typename Request, typename Response>
		friend class RpcReactorUnarySyncHandler;
		template <typename Request, typename Response>
		friend class RpcReactorServerStreaming;
		template <typename Request, typename Response>
		friend class RpcReactorBidiStreaming;
		void init() { shared_self_ = std::shared_ptr<RpcReactorBase>(this); }

		virtual void onFinish()
		{
			// 要先备份一下，否则会造成shared_ptr的析构函数递归调用自己的析构函数
			auto backup = shared_self_;
			shared_self_.reset();
		}

		virtual void handleRequest(RpcEvent& rpc_event) = 0;

		virtual void handleRequestDone(RpcEvent& rpc_event)
		{
		};
		virtual void handleCall(RpcEvent& rpc_event) = 0;

		void handle(RpcEvent rpc_event)
		{
			switch (rpc_event.event_type())
			{
			case RpcEventType::CALL:
				handleCall(rpc_event);
				break;
			case RpcEventType::REQUEST:
				handleRequest(rpc_event);
				break;
			case RpcEventType::REQUEST_DONE:
				handleRequestDone(rpc_event);
				break;
			case RpcEventType::CANCEL:
				onCancel();
				break;
			case RpcEventType::FINISH:
				onFinish();
				break;
			default:
				throw RpcException("internal error: unknown rpc message type");
			}
		}

		void bindRpcStream(RpcEvent& rpc_event)
		{
			this->rpc_stream_rw->bind(rpc_event.rpc_stream());
		}

		TSharedPtr<RpcStreamRW> rpc_stream_rw;
		std::shared_ptr<RpcReactorBase> shared_self_;

	public:
		TSharedPtr<RpcStreamRW> GetRpcStream()
		{
			return rpc_stream_rw;
		}
	};

	template <typename Handler, typename... Args>
	static std::invoke_result_t<Handler, Args...> invokeHandler(Handler&& handler,
	                                                            Args&&... args)
	{
		try
		{
			return std::invoke(std::forward<Handler>(handler),
			                   std::forward<Args>(args)...);
		}
		catch (std::exception& e)
		{
			std::ostringstream oss;
			oss << "handler throwed exception: " << e.what();
			throw RpcException{grpc::StatusCode::INTERNAL, oss.str()};
		} catch (...)
		{
			throw RpcException{
				grpc::StatusCode::INTERNAL,
				"handler throwed unknown exception"
			};
		}
	}

	namespace detail
	{
		template <typename Request>
		class RpcReactorUnaryBase : public RpcReactorBase
		{
		protected:
			// 注意onXXX之类的回调函数以及reactor的析构函数里,绝对不能调用阻塞操作，比如thread.join()，否则消息的处理会被卡住
			// 收到请求消息后调用
			virtual void onRequest(Request& request) = 0;

		private:
			void handleCall(RpcEvent& rpc_event) override
			{
			}

			void handleRequest(RpcEvent& rpc_event) override
			{
				Request request;
				this->rpc_stream_rw->deserialize(request);

				// Grpc Message Debug by WuKunKun:
				if (UTSGrpcMessageDebugSubsystem* DebugSubsystem = UTSGrpcMessageDebugSubsystem::GetInstance())
				{
					// TODO: Add method to serialize?
					// rpc_event.rpc_stream()->method()
					DebugSubsystem->DebugRequest(&request);
				}

				invokeHandler(&RpcReactorUnaryBase::onRequest, *this, request);
			}
		};

		template <typename Response>
		class RpcReactorServerStreamingInterface : virtual public RpcReactorFinisher
		{
		public:
			// 写入一条消息
			virtual void write(const Response& response) = 0;
		};
	} // namespace detail
}
