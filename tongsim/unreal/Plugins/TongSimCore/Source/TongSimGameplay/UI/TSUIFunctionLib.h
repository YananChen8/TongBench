// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "TSUIFunctionLib.generated.h"

struct FGameplayTag;
class UTSActivatableWidget;
/**
 *
 */
UCLASS()
class TONGSIMGAMEPLAY_API UTSUIFunctionLib : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()
public:
	UFUNCTION(BlueprintCallable)
	static UTSActivatableWidget* PushWidgetToLayerForPlayer(UPARAM(meta = (AllowAbstract = false)) TSubclassOf<UTSActivatableWidget> WidgetClass,
		UPARAM(meta = (Categories = "TongSim.UI")) FGameplayTag LayerName);
};
