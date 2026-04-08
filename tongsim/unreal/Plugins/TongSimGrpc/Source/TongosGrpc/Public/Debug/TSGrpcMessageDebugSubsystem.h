#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "TongSimMemoryFixerInterface.h"
#include "TSGrpcLogChannel.h"
#include "TSGrpcMessageDebugSubsystem.generated.h"

// TODO: Debug Interface macro with macro
// #define TONG_SIM_GRPC_MESSAGE_DEBUG 1

UCLASS()
class TONGOSGRPC_API UTSGrpcMessageDebugSubsystem : public UGameInstanceSubsystem
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	static UTSGrpcMessageDebugSubsystem* GetInstance() { return Instance; }

private:
	static UTSGrpcMessageDebugSubsystem* Instance;

	// Debug Log
	uint64 TotalGrpcMessagesLogCount = 0;
	bool bIsLogValid = false;
	int LogMaxLength = 1024;

	// Serialize
	uint64 TotalGrpcMessagesSerializeCount = 0;
	bool bIsSerializeValid = true;

public:
	template <typename RequestType>
	void DebugRequest(const RequestType* Request)
	{
		++TotalGrpcMessagesLogCount;
		if (bIsLogValid && Request)
		{
			FString LogStr;
			RequestLogToString(Request, LogStr);
			if (LogStr.Len() > LogMaxLength)
			{
				LogStr = FString(TEXT("This message length is over %d."), LogMaxLength);
			}
			UE_LOG(LogTongSimGRPC, Log, TEXT("[Grpc Message Request Debug %llu   --   %p]\n%s[~Grpc Message Request Debug %llu   --   %p]\n"),
			       TotalGrpcMessagesLogCount, Request, *LogStr, TotalGrpcMessagesLogCount, Request);
		}
	}

	template <typename RequestType>
	void SerializeGenericRequest(const RequestType* Request, const FString& MethodName)
	{
		++TotalGrpcMessagesSerializeCount;
		if (bIsSerializeValid)
		{
			FString SerializeString;
			RequestSerializeToString(Request, SerializeString);
		}
	}

	template <typename RequestType>
	void SerializeRequest(const RequestType* Request, const FString& ActionName, bool bIsStream)
	{
		++TotalGrpcMessagesSerializeCount;
		if (bIsSerializeValid)
		{
			FString SerializeString;
			RequestSerializeToString(Request, SerializeString);
		}
	}

private:
	/**
	 * @brief
	 * @tparam RequestType
	 * @param Request
	 * @param OutString
	 */
	template <typename RequestType>
	static void RequestLogToString(const RequestType* Request, FString& OutString)
	{
		std::string DebugString;
		if (Request)
		{
			DebugString = Request->DebugString();
		}
		OutString = FString(UTF8_TO_TCHAR(DebugString.c_str()));
		// TongSimDeallocateCString(std::move(DebugString));
	}

	template <typename RequestType>
	static void RequestSerializeToString(const RequestType* Request, FString& OutString)
	{
		std::string SerializeString;
		if (Request)
		{
			Request->SerializePartialToString(&SerializeString);
		}
		OutString = FString(UTF8_TO_TCHAR(SerializeString.c_str()));
		// TongSimDeallocateCString(std::move(SerializeString));
	}
};
