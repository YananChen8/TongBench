#pragma once

namespace tongos
{
	class RpcStream;
	class RpcReactorBase;

	enum class RpcEventType
	{
		// 有新的rpc请求
		CALL,
		// 有新的消息
		REQUEST,
		// 客户端关闭流
		REQUEST_DONE,
		// 客户端断开连接
		CANCEL,
		FINISH,
	};

	class RpcEvent
	{
	public:
		RpcEvent(RpcStream* rpc_stream, RpcReactorBase* rpc_reactor, RpcEventType event_type)
			: rpc_stream_(rpc_stream), backuped_rpc_reactor_(rpc_reactor), event_type_(event_type)
		{
		}

		RpcStream* rpc_stream() { return rpc_stream_; }

		RpcEventType event_type() { return event_type_; }

		RpcReactorBase* backuped_rpc_reactor()
		{
			return backuped_rpc_reactor_;
		}

	private:
		RpcStream* rpc_stream_;
		RpcReactorBase* backuped_rpc_reactor_;
		RpcEventType event_type_;
	};
} // namespace tongos
