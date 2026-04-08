#pragma once
#include "rpc_router.h"
#include "rpc_server_info.h"
#include "rpc_worker.h"
#include <condition_variable>
#include <forward_list>
#include <thread>

namespace tongos
{
	// UE跨模块编译会导致grpc链接出现问题（违反ODR)，从而卡在grpc::InsecureServerCredentials()上
	// class TONGOSGRPC_API RpcServer {
	class TONGOSGRPC_API RpcServer
	{
	public:
		RpcServer(std::string address, RpcRouter& router,
		          std::chrono::milliseconds shutdown_timeout_ms =
			          std::chrono::milliseconds(1000));
		~RpcServer();

		// 添加worker
		void addWorker(RpcWorker::Callback callback);
		// 启动server
		bool start();
		void wait();

	private:
		grpc::ServerBuilder server_builder;
		std::string address;
		RpcServerInfo rpc_server_info;
		std::chrono::milliseconds shutdown_timeout_ms;

		std::unique_ptr<grpc::Server> server;
		std::forward_list<RpcWorker> rpc_workers;
		int index;
	};
} // namespace tongos
