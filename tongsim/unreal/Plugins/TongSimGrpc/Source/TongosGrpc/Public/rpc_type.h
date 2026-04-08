#pragma once
#include <map>
#include <optional>
#include <string>

namespace tongos
{
	enum class RpcType
	{
		UNARY,
		SERVER_STREAMING,
		BIDI_STREAMING,
	};

	class RpcTypeQueryer
	{
	public:
		virtual std::optional<RpcType> queryRpcType(const std::string& method) = 0;
		virtual ~RpcTypeQueryer() = default;
	};
} // namespace tongos
