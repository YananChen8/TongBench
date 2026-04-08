#include "rpc_common.h"

namespace tongos
{
	const ResponseStatus& ResponseStatus::OK = ResponseStatus();
	const ResponseStatus& ResponseStatus::CANCELLED = ResponseStatus(grpc::StatusCode::CANCELLED, "");
}
