// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Engine/LocalPlayer.h"
#include "TSLocalPlayerBase.generated.h"

class APlayerController;
class UObject;
class UTSPrimaryLayout;

/**
 *
 */
UCLASS(config=Engine, transient)
class TONGSIMGAMEPLAY_API UTSLocalPlayerBase : public ULocalPlayer
{
	GENERATED_BODY()

public:
	UTSLocalPlayerBase();

	/** Called when the local player is assigned a player controller */
	DECLARE_MULTICAST_DELEGATE_TwoParams(FPlayerControllerSetDelegate, UTSLocalPlayerBase* LocalPlayer, APlayerController* PlayerController);
	FPlayerControllerSetDelegate OnPlayerControllerSet;

	virtual FString GetGameLoginOptions() const override;

	FDelegateHandle CallAndRegister_OnPlayerControllerSet(FPlayerControllerSetDelegate::FDelegate Delegate);

	UTSPrimaryLayout* GetRootUILayout() const;
};
