#pragma once
#include <queue>
#include <grpcpp/grpcpp.h>

namespace tongos
{
	class TONGOSGRPC_API RpcWriteQueue
	{
	public:
		bool empty();
		void emplace(grpc::ByteBuffer grpc_byte_buffer);
		grpc::ByteBuffer& front();
		void pop();

	private:
		std::queue<grpc::ByteBuffer> queue;
	};
}
