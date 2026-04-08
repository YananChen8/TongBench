#pragma once
#include "rpc_router.h"
#include "rpc_server_info.h"
#include "rpc_stream.h"
#include "util/logger.h"
#include <condition_variable>
#include <functional>
#include <grpcpp/grpcpp.h>
#include <memory>
// #include <semaphore>
#include <thread>

namespace tongos
{
	class RpcWorker
	{
	public:
		using Callback = std::function<void(RpcEvent)>;

		RpcWorker(int index, Callback callback,
		          std::unique_ptr<grpc::ServerCompletionQueue> cq,
		          RpcServerInfo& rpc_server_info)
			: index(index), callback(callback), cq(std::move(cq)),
			  rpc_server_info(rpc_server_info), calls_in_fly(1)
		{
		}

		~RpcWorker()
		{
			// 等待所有处理中的请求都finish了再关闭cq
			// UE中server的关闭和回调处理在一个线程，所以不需要等待
			// {
			//   std::unique_lock lock_guard(shutdown_mu);
			//   while(calls_in_fly != 0){
			//     shutdown_cv.wait(lock_guard);
			//   }
			// }
			tonglog(INFO) << index << " all calls done";
			cq->Shutdown();
			tonglog(INFO) << index << " cq shutdowned";
			worker.join();
			tonglog(INFO) << index << " server worker joined";
		}

		void start() { worker = std::thread(&RpcWorker::work, this); }

	private:
		void work()
		{
			// 开始接收请求
			new RpcStream(&rpc_server_info.generic_service, cq.get(),
			              rpc_server_info.rpc_router);
			tonglog(INFO) << index << " server loop start";

			std::function<void()> increment_calls = [this]() { incrementCalls(); };
			std::function<void()> decrement_calls = [this]() { decrementCalls(); };

			void* tag;
			bool ok;
			while (cq->Next(&tag, &ok))
			{
				tonglog(INFO) << "new tag: " << tag << ", result: " << ok;
				std::optional<RpcEvent> rpc_event =
					RpcStream::handle(tag, ok, increment_calls, decrement_calls);
				if (!rpc_event)
				{
					continue;
				}
				callback(std::move(rpc_event).value());
			}
			tonglog(INFO) << index << " server loop end";
		}

		void incrementCalls()
		{
			calls_in_fly += 1;
			tonglog(DEBUG) << index << " increment calls in fly:" << calls_in_fly;
		}

		void decrementCalls()
		{
			std::scoped_lock lock_guard(shutdown_mu);
			calls_in_fly -= 1;
			tonglog(DEBUG) << index << " decrement calls in fly:" << calls_in_fly;
			if (calls_in_fly == 0)
			{
				shutdown_cv.notify_all();
			}
		}

		int index;
		Callback callback;
		std::unique_ptr<grpc::ServerCompletionQueue> cq;
		RpcServerInfo& rpc_server_info;

		std::thread worker;
		// 平滑关闭
		std::mutex shutdown_mu;
		std::condition_variable shutdown_cv;
		uint64_t calls_in_fly;
	};
} // namespace tongos
