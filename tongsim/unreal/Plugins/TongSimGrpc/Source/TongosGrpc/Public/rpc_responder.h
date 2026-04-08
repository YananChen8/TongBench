#pragma once
#include "rpc_reactor_base.h"

namespace tongos
{
	// template <typename Response> class RpcResponder {
	// public:
	//   RpcResponder(RpcStreamRW &rpc_stream_rw) : rpc_stream_rw(rpc_stream_rw) {}
	//   RpcResponder(const RpcResponder &) = delete;
	//   RpcResponder(RpcResponder &&) = default;
	//   void write(const Response &response) { rpc_stream_rw.write(response); }

	//   void finish(const ResponseStatus &status) { rpc_stream_rw.finish(status); }
	//   ~RpcResponder() {
	//     rpc_stream_rw.finish({grpc::StatusCode::INTERNAL,
	//                           "handler SHOULD provide status by calling finish"});
	//   }
	// private:
	//   RpcStreamRW &rpc_stream_rw;
	// };

	template <typename Response>
	class RpcServerStreamingResponder
	{
	private:
		using Reactor = std::shared_ptr<detail::RpcReactorServerStreamingInterface<Response>>;

	public:
		RpcServerStreamingResponder(Reactor reactor): reactor_(std::move(reactor))
		{
		}

		RpcServerStreamingResponder(const RpcServerStreamingResponder&) = delete;

		RpcServerStreamingResponder(RpcServerStreamingResponder&& other): reactor_(std::move(other.reactor_))
		{
		}

		~RpcServerStreamingResponder()
		{
			if (reactor_)
			{
				reactor_->finish(ResponseStatus{grpc::StatusCode::ABORTED, "server didn't provide a status before leaving"});
			}
		}

		Reactor operator->()
		{
			return reactor_;
		}

	private:
		Reactor reactor_;
	};
} // namespace tongos
