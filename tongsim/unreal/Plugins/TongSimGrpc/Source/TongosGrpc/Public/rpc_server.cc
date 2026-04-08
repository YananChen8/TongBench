#include "rpc_server.h"

namespace tongos
{
	RpcServer::RpcServer(std::string address, RpcRouter& router,
	                     std::chrono::milliseconds shutdown_timeout_ms)
		: address(address), rpc_server_info(router),
		  shutdown_timeout_ms(shutdown_timeout_ms), index(0)
	{
		server_builder.AddListeningPort(address, grpc::InsecureServerCredentials());
		server_builder.RegisterAsyncGenericService(&rpc_server_info.generic_service);
	}

	// 添加worker
	void RpcServer::addWorker(RpcWorker::Callback callback)
	{
		rpc_workers.emplace_front(
			index++, callback, server_builder.AddCompletionQueue(), rpc_server_info);
	}

	// 启动server
	bool RpcServer::start()
	{
		server = server_builder.BuildAndStart();

		if (!server)
		{
			UE_LOG(LogTongSimGRPC, Error, TEXT("[RpcServer::start] Failed to start gRPC server on address: %s"), UTF8_TO_TCHAR(address.c_str()));
			return false;
		}

		for (auto&& worker : rpc_workers)
		{
			worker.start();
		}
		UE_LOG(LogTongSimGRPC, Log, TEXT("[RpcServer::start] Starting gRPC server on address: %s"), UTF8_TO_TCHAR(address.c_str()));
		return true;
	}

	void RpcServer::wait() { server->Wait(); }

	RpcServer::~RpcServer()
	{
		if (server)
		{
			server->Shutdown(std::chrono::system_clock::now() + shutdown_timeout_ms);
			UE_LOG(LogTongSimGRPC, Log, TEXT("[RpcServer::~RpcServer] Server shutdown completed at address: %s"), UTF8_TO_TCHAR(address.c_str()));
		}
	}
} // namespace tongos
