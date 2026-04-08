#pragma once
#include "grpcpp/grpcpp.h"

namespace tongos
{
	class TONGOSGRPC_API ResponseStatus
	{
	public:
		ResponseStatus() : code_(grpc::StatusCode::OK)
		{
		}

		ResponseStatus(grpc::StatusCode code, const std::string& error_message)
			: code_(code), error_message_(error_message)
		{
		}

		ResponseStatus(grpc::StatusCode code, const std::string& error_message,
		               const std::string& error_details)
			: code_(code),
			  error_message_(error_message),
			  binary_error_details_(error_details)
		{
		}

		grpc::StatusCode error_code() const { return code_; }
		/// Return the instance's error message.
		const std::string& error_message() const { return error_message_; }
		/// Return the (binary) error details.
		// Usually it contains a serialized google.rpc.Status proto.
		const std::string& error_details() const { return binary_error_details_; }

		grpc::Status to_grpc_status() const
		{
			return {code_, error_message_, binary_error_details_};
		}

		/// Is the status OK?
		bool ok() const { return code_ == grpc::StatusCode::OK; }

		// Ignores any errors. This method does nothing except potentially suppress
		// complaints from any tools that are checking that errors are not dropped on
		// the floor.
		void IgnoreError() const
		{
		}

		// Pre-defined special status objects.
		/// An OK pre-defined instance.
		static const ResponseStatus& OK;
		/// A CANCELLED pre-defined instance.
		static const ResponseStatus& CANCELLED;

	private:
		grpc::StatusCode code_;
		std::string error_message_;
		std::string binary_error_details_;
	};
}
