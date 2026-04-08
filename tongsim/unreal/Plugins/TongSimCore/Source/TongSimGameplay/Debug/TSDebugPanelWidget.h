// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UI/Layer/TSActivatableWidget.h"
#include "TSDebugPanelWidget.generated.h"

class UTSWindowBase;
/**
 *
 */
UCLASS()
class TONGSIMGAMEPLAY_API UTSDebugPanelWidget : public UTSActivatableWidget
{
	GENERATED_BODY()

public:
	UTSWindowBase* AddNewDebugWindow(TSubclassOf<UTSWindowBase> WindowClass, TSubclassOf<UUserWidget> ChildWidgetClass, const FVector2D StartSize = FVector2D(640.f, 480.f), const bool bCenterToScreen = true, const bool bCanDrag = true, const bool bCanResize = true);

private:
	UPROPERTY(meta = (BindWidget))
	class UCanvasPanel* MainPanel;

	UPROPERTY(Transient)
	TSet<TObjectPtr<UTSWindowBase>> DebugWindows;
};
