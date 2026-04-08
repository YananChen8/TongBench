// Fill out your copyright notice in the Description page of Project Settings.


#include "TSLocalPlayerBase.h"

#include "Common/TongSimGameplayFunc.h"
#include "Engine/GameInstance.h"
#include "GameFramework/PlayerController.h"

UTSLocalPlayerBase::UTSLocalPlayerBase() : Super(FObjectInitializer::Get())
{
}

FString UTSLocalPlayerBase::GetGameLoginOptions() const
{
	FURL URL(nullptr, TEXT(""), TRAVEL_Absolute);

	/* Platform */
	if (UTongSimGameplayFunc::IsHMDConnected())
	{
		URL.AddOption(TEXT("PlayerType=VR"));
	}
	else
	{
		URL.AddOption(TEXT("PlayerType=HumanPlayer"));
	}

	return URL.ToString();
}

FDelegateHandle UTSLocalPlayerBase::CallAndRegister_OnPlayerControllerSet(FPlayerControllerSetDelegate::FDelegate Delegate)
{
	APlayerController* PC = GetPlayerController(GetWorld());

	if (PC)
	{
		Delegate.Execute(this, PC);
	}

	return OnPlayerControllerSet.Add(Delegate);
}

UTSPrimaryLayout* UTSLocalPlayerBase::GetRootUILayout() const
{
	// TODO by wukunlun:
	return nullptr;
}
