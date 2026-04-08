// Fill out your copyright notice in the Description page of Project Settings.


#include "TSPlayerControllerBase.h"
#include "EnhancedInputSubsystems.h"
#include "InputMappingContext.h"
#include "TSLocalPlayerBase.h"
#include "TSLogChannels.h"
#include "UserSettings/EnhancedInputUserSettings.h"

ATSPlayerControllerBase::ATSPlayerControllerBase(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer)
{
}

void ATSPlayerControllerBase::ReceivedPlayer()
{
	Super::ReceivedPlayer();

	if (UTSLocalPlayerBase* LocalPlayer = Cast<UTSLocalPlayerBase>(Player))
	{
		LocalPlayer->OnPlayerControllerSet.Broadcast(LocalPlayer, this);
	}
}

void ATSPlayerControllerBase::SetPlayerMappableInputConfig(TSoftObjectPtr<UInputMappingContext> InputMapping) const
{
	UEnhancedInputLocalPlayerSubsystem* Subsystem = ULocalPlayer::GetSubsystem<UEnhancedInputLocalPlayerSubsystem>(
		GetLocalPlayer());
	check(Subsystem);

	if (UInputMappingContext* IMC = InputMapping.LoadSynchronous())
	{
		Subsystem->ClearAllMappings();
		if (UEnhancedInputUserSettings* Settings = Subsystem->GetUserSettings())
		{
			Settings->RegisterInputMappingContext(IMC);
		}
		FModifyContextOptions Options = {};
		Options.bIgnoreAllPressedKeysUntilRelease = false;
		// Actually add the input mapping to the local player
		Subsystem->AddMappingContext(IMC, 0, Options);
	}
	else
	{
		UE_LOG(LogTongSimCore, Error, TEXT("PlayerController got null InputMapping."));
	}
}
