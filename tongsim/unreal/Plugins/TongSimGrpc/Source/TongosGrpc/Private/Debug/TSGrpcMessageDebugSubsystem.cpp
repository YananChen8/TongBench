#include "Debug/TSGrpcMessageDebugSubsystem.h"
#include "Debug/TSGrpcSettings.h"

UTSGrpcMessageDebugSubsystem* UTSGrpcMessageDebugSubsystem::Instance = nullptr;

void UTSGrpcMessageDebugSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	UGameInstanceSubsystem::Initialize(Collection);

	if (const UTSGrpcSettings* GameSettings = GetDefault<UTSGrpcSettings>())
	{
		bIsLogValid = GameSettings->bDebugGrpcMessage;
		LogMaxLength = GameSettings->MaxGrpcMessageLogLength;
		bIsSerializeValid = GameSettings->bSerializeGrpcMessage;
	}

	Instance = this;
}

void UTSGrpcMessageDebugSubsystem::Deinitialize()
{
	bIsLogValid = false;
	Instance = nullptr;
	UGameInstanceSubsystem::Deinitialize();
}
