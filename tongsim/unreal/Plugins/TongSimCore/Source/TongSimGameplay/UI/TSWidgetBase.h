// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "TSWidgetBase.generated.h"

/**
 *
 */
UCLASS(ClassGroup = UI, meta = (Category = "TongSim|UI", DisableNativeTick))
class TONGSIMGAMEPLAY_API UTSWidgetBase : public UUserWidget
{
	GENERATED_BODY()

protected:
	template <typename PlayerControllerT = APlayerController>
	PlayerControllerT& GetOwningPlayerChecked() const
	{
		PlayerControllerT* PC = GetOwningPlayer<PlayerControllerT>();
		check(PC);
		return *PC;
	}

	TSharedPtr<FSlateUser> GetOwnerSlateUser() const;
};
