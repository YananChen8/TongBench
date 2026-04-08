
#pragma once

#include "CoreMinimal.h"
#include "Engine/DeveloperSettings.h"
#include "TSGrpcSettings.generated.h"

/**
 *
 */
UCLASS(config=TongSimGrpc, defaultconfig, MinimalAPI, meta=(DisplayName="TongSim gRpc Setting"))
class UTSGrpcSettings : public UDeveloperSettings
{
	GENERATED_BODY()

public:
	UPROPERTY(Config, EditAnywhere, Category="TongSim|Debug")
	bool bDebugGrpcMessage = false;

	UPROPERTY(Config, EditAnywhere, Category="TongSim|Debug")
	int MaxGrpcMessageLogLength = 1024;

	UPROPERTY(Config, EditAnywhere, Category="TongSim|Debug")
	bool bSerializeGrpcMessage = false;

};
