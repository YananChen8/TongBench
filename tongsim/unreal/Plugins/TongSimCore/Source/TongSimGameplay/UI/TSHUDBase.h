// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/HUD.h"
#include "TSHUDBase.generated.h"

class UTSDebugPanelWidget;
class UTSWindowBase;
/*
 * Only For Debug ?
 */
UCLASS()
class TONGSIMGAMEPLAY_API ATSHUDBase : public AHUD
{
	GENERATED_BODY()

public:
	ATSHUDBase();

	UFUNCTION(BlueprintCallable, BlueprintCosmetic, Category = "TongSim|UI")
	UTSWindowBase* CreateDebugWindowWithChildWidget(TSubclassOf<UUserWidget> ChildWidgetClass, const FVector2D StartSize = FVector2D(640.f, 480.f), const FText& WindowTitle = FText(),
	                                                const bool bCenterToScreen = true, const bool bCanDrag = true, const bool bCanResize = true);

protected:
	virtual void BeginPlay() override;

private:
	/*
	 * Debug Panel Widget
	 */

	UPROPERTY(EditDefaultsOnly, Category = "TongSim|Debug", meta = (AllowPrivateAccess))
	TSubclassOf<UTSDebugPanelWidget> DebugPanelWidgetClass;

	TWeakObjectPtr<UTSDebugPanelWidget> DebugPanelWidget;

	UPROPERTY(EditDefaultsOnly, Category = "TongSim|Debug", meta = (AllowPrivateAccess))
	TSubclassOf<UTSWindowBase> DebugWindowClass;

	void InitDebugPanel();
};
