// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameplayTagContainer.h"
#include "GameFramework/WorldSettings.h"
#include "TSWorldSettings.generated.h"

/**
 * The default world settings object, used primarily to set the default gameplay experience to use when playing on this map
 */
UCLASS()
class TONGSIMGAMEPLAY_API ATSWorldSettings : public AWorldSettings
{
	GENERATED_BODY()

public:
	ATSWorldSettings(const FObjectInitializer& ObjectInitializer);
};
