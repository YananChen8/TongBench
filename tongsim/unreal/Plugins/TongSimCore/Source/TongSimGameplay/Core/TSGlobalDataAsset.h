// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "TSGlobalDataAsset.generated.h"

/**
 *	Non-mutable data asset that contains global game data.
 */
UCLASS(BlueprintType, Const, Meta = (DisplayName = "TongSim Global Game Data", ShortTooltip = "Data asset containing global game data."))
class UTSGlobalDataAsset : public UPrimaryDataAsset
{
	GENERATED_BODY()

	UTSGlobalDataAsset();

	static const UTSGlobalDataAsset & Get();

	virtual FPrimaryAssetId GetPrimaryAssetId() const override;
public:
	UPROPERTY(EditDefaultsOnly, Category = "TongSim", meta = (DisplayName = "Sync Loaded assets"))
	TArray<TSoftClassPtr<UObject>> SyncLoadAssets;

	UPROPERTY(EditDefaultsOnly, Category = "TongSim", meta = (DisplayName = "Async Loaded assets"))
	TArray<TSoftClassPtr<UObject>> AsyncLoadAssets;
};
