#include "rpc_write_queue.h"

namespace tongos
{
	bool RpcWriteQueue::empty()
	{
		return queue.empty();
	}

	void RpcWriteQueue::emplace(grpc::ByteBuffer grpc_byte_buffer)
	{
		queue.emplace(std::move(grpc_byte_buffer));
	}

	grpc::ByteBuffer& RpcWriteQueue::front()
	{
		return queue.front();
	}

	void RpcWriteQueue::pop()
	{
		queue.pop();
	}
}
