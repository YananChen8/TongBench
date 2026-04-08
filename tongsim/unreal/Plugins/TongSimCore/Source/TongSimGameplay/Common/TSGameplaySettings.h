// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Engine/DeveloperSettings.h"
#include "TSGameplaySettings.generated.h"

class UTSPrimaryLayout;
class ACharacter;
/**
 *
 */
UCLASS(config=TongSimGameplay, defaultconfig, MinimalAPI, meta=(DisplayName="TongSim Gameplay Setting"))
class UTSGameplaySettings : public UDeveloperSettings
{
	GENERATED_BODY()

public:
	UPROPERTY(Config, EditAnywhere, Category="TongSim|UI")
	TSoftClassPtr<UTSPrimaryLayout> LayoutClass;

	UPROPERTY(Config, EditAnywhere, Category="TongSim|NPC")
	TSoftClassPtr<ACharacter> NPCClass;

	UPROPERTY(Config, EditAnywhere, Category="TongSim|Distribution")
	int MaxAgentNumber = 1;

	UPROPERTY(Config, EditAnywhere, Category="TongSim|Distribution")
	float AsyncGrpcDeadline = 18.f;

	UPROPERTY(Config, EditAnywhere, Category="TongSim|Debug")
	bool bLogPerformance = false;

	UPROPERTY(Config, EditAnywhere, Category="TongSim|Debug")
	float LogPerformanceIntervalInSec = 10.f;

	UPROPERTY(Config, EditAnywhere, Category="TongSim|Image")
	float DepthBufferCoefficient_FakeCameraVisibility = 0.1f;

	UPROPERTY(Config, EditAnywhere, Category="TongSim|Image")
	bool bCullingWithLineTrace = false;
};
