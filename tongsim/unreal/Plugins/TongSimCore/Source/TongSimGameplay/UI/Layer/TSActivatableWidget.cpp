// Fill out your copyright notice in the Description page of Project Settings.


#include "TSActivatableWidget.h"

#include "TSLogChannels.h"

void UTSActivatableWidget::ActivateWidget()
{
	if (!bIsActive)
	{
		UE_LOG(LogTongSimCore, Log, TEXT("[%s] widget -> Activated"), *GetName());
		bIsActive = true;
		NativeOnActivated();
	}
}

void UTSActivatableWidget::DeactivateWidget()
{
	if (bIsActive)
	{
		UE_LOG(LogTongSimCore, Log, TEXT("[%s] widget -> Deactivated"), *GetName());
		bIsActive = false;
		NativeOnDeactivated();
	}
}

void UTSActivatableWidget::NativeConstruct()
{
	Super::NativeConstruct();

	if (bAutoActivate)
	{
		UE_LOG(LogTongSimCore, Log, TEXT("[%s] Widget auto-activated"), *GetName());
		ActivateWidget();
	}
}

void UTSActivatableWidget::NativeDestruct()
{
	if (UGameInstance* GameInstance = GetGameInstance<UGameInstance>())
	{
		// Deactivations might rely on members of the game instance to validly run.
		// If there's no game instance, any cleanup done in Deactivation will be irrelevant; we're shutting down the game
		DeactivateWidget();
	}
	Super::NativeDestruct();
}

void UTSActivatableWidget::NativeOnActivated()
{
	if (ensureMsgf(bIsActive, TEXT("[%s] has called NativeOnActivated, but isn't actually activated! Never call this directly - call ActivateWidget()"), *GetName()))
	{
		if (bSetVisibilityOnActivated)
		{
			SetVisibility(ActivatedVisibility);
		}

		OnActivated().Broadcast();
	}
}

void UTSActivatableWidget::NativeOnDeactivated()
{
	if (ensure(!bIsActive))
	{
		if (bSetVisibilityOnDeactivated)
		{
			SetVisibility(DeactivatedVisibility);
		}
		OnDeactivated().Broadcast();
	}
}
