#pragma once
#include "chrono"
#include "rpc_router.h"

namespace tongos
{
	class RpcServerInfo
	{
	public:
		RpcServerInfo(RpcRouter& rpc_router) : rpc_router(rpc_router)
		{
		}

	private:
		friend class RpcWorker;
		friend class RpcServer;
		RpcRouter& rpc_router;
		grpc::AsyncGenericService generic_service;
	};
} // namespace tongos
