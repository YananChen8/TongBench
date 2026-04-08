// Fill out your copyright notice in the Description page of Project Settings.


#include "TSWindowBase.h"

#include "TSButtonBase.h"
#include "Blueprint/WidgetLayoutLibrary.h"
#include "Blueprint/SlateBlueprintLibrary.h"
#include "Blueprint/WidgetBlueprintLibrary.h"
#include "TSLogChannels.h"
#include "Components/Border.h"
#include "Components/CanvasPanelSlot.h"
#include "Components/CanvasPanel.h"
#include "Components/TextBlock.h"
#include "Kismet/GameplayStatics.h"

UTSWindowBase::UTSWindowBase(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer)
{
	ResizeMinWidth = 400.f;
	ResizeMaxWidth = 0.f;

	ResizeMinHeight = 400.f;
	ResizeMaxHeight = 0.f;

	bEnableDrag = bEnableResizing = true;
	bIsMouseButtonDown = bIsDragging = bIsResizing = bIsAlignmentAccountedFor = bStartInCenterScreen = false;
	LastMousePosition = PreResizeAlignment = PreResizeOffset = PreDragSize = StartSize = FVector2D::ZeroVector;

	DragKey = FKey(FName("LeftMouseButton"));

	CurrentZOrder = 1;

	SetIsFocusable(true);
}

UTSWindowBase* UTSWindowBase::CreateDraggableWindow(const UObject* WorldContextObject, TSubclassOf<UTSWindowBase> WindowClass, TSubclassOf<UUserWidget> ChildWidgetClass, const FVector2D StartSize, const bool bCenterToScreen, const bool bCanDrag, const bool bCanResize)
{
	if (WindowClass)
	{
		APlayerController* MyPlayerController = UGameplayStatics::GetPlayerController(WorldContextObject, 0);
		UUserWidget* ProxyChild = nullptr;

		if (ChildWidgetClass)
		{
			ProxyChild = Cast<UUserWidget>(CreateWidget(MyPlayerController, ChildWidgetClass));
		}

		return CreateDraggableWindowFromWidget(WorldContextObject, WindowClass, ProxyChild, StartSize, bCenterToScreen, bCanDrag, bCanResize);
	}

	UE_LOG(LogTongSimCore, Error, TEXT("No window class specified. No window was created."));
	return nullptr;
}

UTSWindowBase* UTSWindowBase::CreateDraggableWindowFromWidget(const UObject* WorldContextObject, TSubclassOf<UTSWindowBase> WindowClass, UUserWidget* NewChildWidget, const FVector2D StartSize, const bool bCenterToScreen, const bool bCanDrag, const bool bCanResize)
{
	if (WindowClass)
	{
		APlayerController* MyPlayerController = UGameplayStatics::GetPlayerController(WorldContextObject, 0);
		UTSWindowBase* ProxyWindow = Cast<UTSWindowBase>(CreateWidget(MyPlayerController, WindowClass));
		ProxyWindow->StartSize = StartSize;
		ProxyWindow->bStartInCenterScreen = bCenterToScreen;
		ProxyWindow->bEnableDrag = bCanDrag;
		ProxyWindow->bEnableResizing = bCanResize;
		if (NewChildWidget)
		{
			ProxyWindow->AddContentWidget(NewChildWidget);
		}
		return ProxyWindow;
	}

	UE_LOG(LogTongSimCore, Error, TEXT("No window class specified. No window was created."));
	return nullptr;
}

void UTSWindowBase::SetWindowTile(FText InText)
{
	if (WindowTitle)
	{
		WindowTitle->SetText(InText);
	}
}

void UTSWindowBase::NativeConstruct()
{
	ParentSlot = UWidgetLayoutLibrary::SlotAsCanvasSlot(this);

	if (CloseButton)
	{
		CloseButton->OnClicked().AddLambda([this]()
		{
			this->RemoveFromParent();
		});
	}

	if (ParentSlot == nullptr)
	{
		UE_LOG(LogTongSimCore, Error, TEXT("TS window requires its parent to be Canvas panel"));
	}
	else
	{
		UpdateWindowSize(StartSize);

		if (bStartInCenterScreen)
		{
			CenterWindowToScreen();
		}

		if (WindowTitleBorderWidget)
		{
			WindowTitleBorderWidget->OnMouseButtonUpEvent.BindUFunction(this, FName("Internal_OnMouseButtonUp_WindowTitleBorder"));
			WindowTitleBorderWidget->OnMouseButtonDownEvent.BindUFunction(this, FName("Internal_OnMouseButtonDown_WindowTitleBorder"));
		}
		else
		{
			UE_LOG(LogTongSimCore, Error, TEXT("Window Title Border was not found. Make sure you have a 'Border' widget with 'Is Variable' enabled and name set to 'WindowTitleBorderWidget'. This will act as the title bar where you can click and drag the window."));
		}

		if (bEnableResizing)
		{
			if (ResizeAreaWidget == nullptr)
			{
				UE_LOG(LogTongSimCore, Error, TEXT("Resizing was enabled but Resize Area Widget was not found. Make sure you have a 'Border' widget with 'Is Variable' enabled and name set to 'ResizeAreaWidget'."));
			}
			else
			{
				ResizeAreaWidget->OnMouseButtonDownEvent.BindUFunction(this, FName("Internal_OnMouseButtonDown_ResizeArea"));
			}
		}
	}

	Super::NativeConstruct();
}

FReply UTSWindowBase::NativeOnMouseMove(const FGeometry& InGeometry, const FPointerEvent& InMouseEvent)
{
	Super::NativeOnMouseMove(InGeometry, InMouseEvent);

	if (bIsMouseButtonDown && ParentSlot)
	{
		FVector2D OutPixelPosition, OutViewportPosition;
		USlateBlueprintLibrary::AbsoluteToViewport(this, InMouseEvent.GetScreenSpacePosition(), OutPixelPosition, OutViewportPosition);
		const FVector2D ViewportSize = UWidgetLayoutLibrary::GetViewportSize(this);
		const bool bIsMouseOffScreen = OutPixelPosition.X < 5.f || OutPixelPosition.Y < 5.f || OutPixelPosition.X > (ViewportSize.X - 5.f) || OutPixelPosition.Y > (ViewportSize.Y - 5.f);
		if (bIsMouseOffScreen)
		{
			Internal_OnMouseButtonUpEvent();
			return FReply::Handled();
		}

		USlateBlueprintLibrary::AbsoluteToViewport(this, InMouseEvent.GetScreenSpacePosition(), OutPixelPosition, OutViewportPosition);
		FVector2D MouseDelta = OutViewportPosition - LastMousePosition;
		FEventReply EventReply = UWidgetBlueprintLibrary::DetectDragIfPressed(InMouseEvent, this, DragKey);
		if (bIsDragging)
		{
			ParentSlot->SetPosition(ParentSlot->GetPosition() + MouseDelta);
		}
		else if (bIsResizing)
		{
			if (bIsAlignmentAccountedFor)
			{
				const FVector2D RequestedSize = Internal_DetermineNewSize(MouseDelta);
				ParentSlot->SetSize(RequestedSize);
			}
			else
			{
				ParentSlot->SetAlignment(FVector2D::ZeroVector);
				ParentSlot->SetPosition(ParentSlot->GetPosition() - PreResizeOffset);
				bIsAlignmentAccountedFor = true;
				return FReply::Handled();
			}
		}
		else
		{
			return UWidgetBlueprintLibrary::CaptureMouse(EventReply, this).NativeReply;
		}

		USlateBlueprintLibrary::AbsoluteToViewport(this, InMouseEvent.GetScreenSpacePosition(), OutPixelPosition, LastMousePosition);
		return UWidgetBlueprintLibrary::CaptureMouse(EventReply, this).NativeReply;
	}

	return FReply::Handled();
}

FReply UTSWindowBase::NativeOnMouseButtonUp(const FGeometry& InGeometry, const FPointerEvent& InMouseEvent)
{
	Super::NativeOnMouseButtonUp(InGeometry, InMouseEvent);
	Internal_OnMouseButtonUpEvent();
	FEventReply EventReply = UWidgetBlueprintLibrary::DetectDragIfPressed(InMouseEvent, this, DragKey);
	return UWidgetBlueprintLibrary::ReleaseMouseCapture(EventReply).NativeReply;
}

void UTSWindowBase::AddContentWidgetClass(TSubclassOf<UUserWidget> InWidgetClass, const bool bClearPrevious)
{
	if (InWidgetClass)
	{
		APlayerController* MyPlayerController = UGameplayStatics::GetPlayerController(this, 0);
		AddContentWidget(CreateWidget(MyPlayerController, InWidgetClass), bClearPrevious);
	}
}

void UTSWindowBase::AddContentWidget(UUserWidget* InWidget, const bool bClearPrevious)
{
	if (ChildWidget && bClearPrevious)
	{
		ChildWidget->RemoveFromParent();
		ChildWidget = nullptr;
	}

	ChildWidget = InWidget;
	Internal_AddContentWidget(bClearPrevious);
}

void UTSWindowBase::CenterWindowToScreen()
{
	ParentSlot->SetAnchors(FAnchors(0.5));
	ParentSlot->SetAlignment(FVector2D(0.5));
	ParentSlot->SetPosition(FVector2D::ZeroVector);
}

bool UTSWindowBase::UpdateWindowSize(const FVector2D& InNewSize)
{
	if (InNewSize != FVector2D::ZeroVector)
	{
		ParentSlot->SetSize(InNewSize); //A1
		return true;
	}

	return false;
}

bool UTSWindowBase::GetChildWidget(UUserWidget*& OutChildWidget) const
{
	OutChildWidget = ChildWidget;
	return OutChildWidget != nullptr;
}

void UTSWindowBase::Internal_AddContentWidget(const bool bClearPrevious)
{
	if (bClearPrevious)
	{
		ChildWidgetCanvas->ClearChildren();
	}

	if (ChildWidget)
	{
		ChildWidgetCanvas->AddChildToCanvas(ChildWidget);
		// K2_OnContentWidgetAdded(ChildWidget);

		if (UCanvasPanelSlot* ChildWidgetRootPanelSlot = UWidgetLayoutLibrary::SlotAsCanvasSlot(ChildWidget))
		{
			ChildWidgetRootPanelSlot->SetAnchors(FAnchors(0.f, 0.f, 1.f, 1.f));
			ChildWidgetRootPanelSlot->SetOffsets(FMargin(0.f));
		}
		else
		{
			UE_LOG(LogTongSimCore, Warning, TEXT("TongSim Window Child Widget %s cant find root canvas slot."), *GetNameSafe(ChildWidget))
		}
	}
}

void UTSWindowBase::Internal_OnMouseButtonUpEvent()
{
	if (bIsResizing && bIsAlignmentAccountedFor && ParentSlot)
	{
		const FVector2D SizeDifference = ParentSlot->GetSize() - PreDragSize;
		const FVector2D NewPosition = (SizeDifference * PreResizeAlignment) + PreResizeOffset + (ParentSlot->GetPosition());
		ParentSlot->SetPosition(NewPosition);
		ParentSlot->SetAlignment(PreResizeAlignment);
	}

	if (bIsDragging)
	{
		K2_OnDragStop();
	}

	bIsAlignmentAccountedFor = false;
	bIsMouseButtonDown = false;
	bIsDragging = false;
	bIsResizing = false;
}

const FVector2D UTSWindowBase::Internal_DetermineNewSize(const FVector2D& InDelta) const
{
	if (ParentSlot)
	{
		const FVector2D Local_Original = ParentSlot->GetSize();
		const float Local_OriginalX = Local_Original.X;
		const float Local_OriginalY = Local_Original.Y;
		const float Local_DeltaX = InDelta.X;
		const float Local_DeltaY = InDelta.Y;

		float TempWidth = 0.f;
		float TempHeight = 0.f;

		if (ResizeMaxWidth > 0.f)
		{
			const float Local_ValueToClamp = Local_OriginalX + Local_DeltaX;
			TempWidth = FMath::Clamp(Local_ValueToClamp, ResizeMinWidth, ResizeMaxWidth);
		}
		else
		{
			TempWidth = FMath::Max<float>((Local_OriginalX + Local_DeltaX), ResizeMinWidth);
		}

		if (ResizeMaxHeight > 0.f)
		{
			const float Local_ValueToClamp = Local_OriginalY + Local_DeltaY;
			TempHeight = FMath::Clamp(Local_ValueToClamp, ResizeMinHeight, ResizeMaxHeight);
		}
		else
		{
			TempHeight = FMath::Max<float>((Local_OriginalY + Local_DeltaY), ResizeMinHeight);
		}

		return FVector2D(TempWidth, TempHeight);
	}

	return FVector2D::ZeroVector;
}

FEventReply UTSWindowBase::Internal_OnMouseButtonUp_WindowTitleBorder(FGeometry InGeometry, const FPointerEvent& InMouseEvent)
{
	FEventReply EventReply = FEventReply();
	EventReply.NativeReply = NativeOnMouseButtonUp(InGeometry, InMouseEvent);
	return EventReply;
}

FEventReply UTSWindowBase::Internal_OnMouseButtonDown_WindowTitleBorder(FGeometry InGeometry, const FPointerEvent& InMouseEvent)
{
	if (bEnableDrag)
	{
		Internal_OnMouseButtonUp_WindowTitleBorder(InGeometry, InMouseEvent);
		bIsMouseButtonDown = true;
		bIsDragging = true;

		FVector2D OutPixelPosition;
		USlateBlueprintLibrary::AbsoluteToViewport(this, InMouseEvent.GetScreenSpacePosition(), OutPixelPosition, LastMousePosition);
		FEventReply EventReply = UWidgetBlueprintLibrary::DetectDragIfPressed(InMouseEvent, this, DragKey);
		K2_OnDragStart(InMouseEvent);
		return UWidgetBlueprintLibrary::CaptureMouse(EventReply, this);
	}

	return FEventReply();
}

FEventReply UTSWindowBase::Internal_OnMouseButtonDown_ResizeArea(FGeometry InGeometry, const FPointerEvent& InMouseEvent)
{
	bIsMouseButtonDown = true;
	bIsResizing = true;

	FEventReply EventReply = FEventReply();
	if (ParentSlot)
	{
		FVector2D OutPixelPosition;
		USlateBlueprintLibrary::AbsoluteToViewport(this, InMouseEvent.GetScreenSpacePosition(), OutPixelPosition, LastMousePosition);
		PreResizeAlignment = ParentSlot->GetAlignment();
		PreDragSize = ParentSlot->GetSize();
		PreResizeOffset = PreDragSize * PreResizeAlignment;
		EventReply = UWidgetBlueprintLibrary::DetectDragIfPressed(InMouseEvent, this, DragKey);
		// K2_OnResizeStart(InMouseEvent);
	}

	return UWidgetBlueprintLibrary::CaptureMouse(EventReply, this);
}
