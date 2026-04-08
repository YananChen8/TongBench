// Fill out your copyright notice in the Description page of Project Settings.


#include "TSDebugPanelWidget.h"

#include "Components/CanvasPanel.h"
#include "UI/Common/TSWindowBase.h"

UTSWindowBase* UTSDebugPanelWidget::AddNewDebugWindow(TSubclassOf<UTSWindowBase> WindowClass, TSubclassOf<UUserWidget> ChildWidgetClass, const FVector2D StartSize, const bool bCenterToScreen, const bool bCanDrag, const bool bCanResize)
{
	if (!IsValid(MainPanel))
	{
		return nullptr;
	}

	if (const APlayerController* PC = GetOwningPlayer())
	{
		if (UTSWindowBase* NewDebugWindow = UTSWindowBase::CreateDraggableWindow(PC, WindowClass, ChildWidgetClass, StartSize, bCenterToScreen, bCanDrag, bCanResize))
		{
			MainPanel->AddChildToCanvas(NewDebugWindow);
			DebugWindows.Emplace(NewDebugWindow);
			return NewDebugWindow;
		}
	}
	return nullptr;
}
