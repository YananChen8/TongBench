// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UI/TSWidgetBase.h"
#include "TSWindowBase.generated.h"

/**
 *
 */
UCLASS(BlueprintType, Blueprintable, ClassGroup = UI, meta = (Category = "TongSim|UI"))
class TONGSIMGAMEPLAY_API UTSWindowBase : public UTSWidgetBase
{
	GENERATED_BODY()

public:
	UTSWindowBase(const FObjectInitializer& ObjectInitializer);

	UFUNCTION(BlueprintCallable, BlueprintCosmetic, Category = "TongSim|UI", DisplayName = "Create TongSim|UI (Child Widget Class)", meta = (WorldContext = "WorldContextObject"))
	static UTSWindowBase* CreateDraggableWindow(const UObject* WorldContextObject, TSubclassOf<UTSWindowBase> WindowClass, TSubclassOf<UUserWidget> ChildWidgetClass, const FVector2D StartSize = FVector2D(640.f, 480.f), const bool bCenterToScreen = true, const bool bCanDrag = true, const bool bCanResize = true);

	UFUNCTION(BlueprintCallable, BlueprintCosmetic, Category = "TongSim|UI", DisplayName = "Create TongSim|UI (Child Widget Reference)", meta = (WorldContext = "WorldContextObject"))
	static UTSWindowBase* CreateDraggableWindowFromWidget(const UObject* WorldContextObject, TSubclassOf<UTSWindowBase> WindowClass, UUserWidget* NewChildWidget, const FVector2D StartSize = FVector2D(640.f, 480.f), const bool bCenterToScreen = true, const bool bCanDrag = true, const bool bCanResize = true);

protected:
	virtual void NativeConstruct() override;
	virtual FReply NativeOnMouseMove(const FGeometry& InGeometry, const FPointerEvent& InMouseEvent) override;
	virtual FReply NativeOnMouseButtonUp(const FGeometry& InGeometry, const FPointerEvent& InMouseEvent) override;

	UFUNCTION(BlueprintCallable, BlueprintCosmetic, Category = "TongSim|UI")
	void AddContentWidgetClass(TSubclassOf<class UUserWidget> InWidgetClass, const bool bClearPrevious = true);

	UFUNCTION(BlueprintCallable, BlueprintCosmetic, Category = "TongSim|UI")
	void AddContentWidget(class UUserWidget* InWidget, const bool bClearPrevious = true);

	UFUNCTION(BlueprintCallable, BlueprintCosmetic, Category = "TongSim|UI")
	void CenterWindowToScreen();

	UFUNCTION(BlueprintCallable, BlueprintCosmetic, Category = "TongSim|UI")
	bool UpdateWindowSize(const FVector2D& InNewSize);

	UFUNCTION(BlueprintPure, BlueprintCosmetic, Category = "TongSim|UI")
	bool GetChildWidget(UUserWidget*& OutChildWidget) const;

	/*  */
	UFUNCTION(BlueprintImplementableEvent, BlueprintCosmetic, Category = "TongSim|UI", DisplayName = "On Drag Start")
	void K2_OnDragStart(const FPointerEvent& InMouseEvent);

	UFUNCTION(BlueprintImplementableEvent, BlueprintCosmetic, Category = "TongSim|UI", DisplayName = "On Drag Stop")
	void K2_OnDragStop();

private:
	void Internal_AddContentWidget(const bool bClearPrevious);
	void Internal_OnMouseButtonUpEvent();
	const FVector2D Internal_DetermineNewSize(const FVector2D& InDelta) const;

	UFUNCTION()
	FEventReply Internal_OnMouseButtonUp_WindowTitleBorder(FGeometry InGeometry, const FPointerEvent& InMouseEvent);

	UFUNCTION()
	FEventReply Internal_OnMouseButtonDown_WindowTitleBorder(FGeometry InGeometry, const FPointerEvent& InMouseEvent);

	UFUNCTION()
	FEventReply Internal_OnMouseButtonDown_ResizeArea(FGeometry InGeometry, const FPointerEvent& InMouseEvent);

	/* Allows you to drag this window. */
	UPROPERTY(EditAnywhere, Category = "TongSim|UI")
	uint8 bEnableDrag : 1;

	/* Allows you to resize this window. */
	UPROPERTY(EditAnywhere, Category = "TongSim|UI")
	uint8 bEnableResizing : 1;

	/* Minimum width (in pixels) you can resize to. */
	UPROPERTY(EditAnywhere, Category = "TongSim|UI", meta = (EditCondition = "bEnableResizing"))
	float ResizeMinWidth;

	/* Maximum width (in pixels) you can resize to. */
	UPROPERTY(EditAnywhere, Category = "TongSim|UI", meta = (EditCondition = "bEnableResizing"))
	float ResizeMaxWidth;

	/* Minimum height (in pixels) you can resize to. */
	UPROPERTY(EditAnywhere, Category = "TongSim|UI", meta = (EditCondition = "bEnableResizing"))
	float ResizeMinHeight;

	/* Maximum height (in pixels) you can resize to. */
	UPROPERTY(EditAnywhere, Category = "TongSim|UI", meta = (EditCondition = "bEnableResizing"))
	float ResizeMaxHeight;

	/* Key used to drag or resize window. Defaults to Left Mouse Button (LMB) because it makes sense. */
	UPROPERTY(EditAnywhere, AdvancedDisplay, Category = "TongSim|UI", meta = (EditCondition = "bEnableDrag && bEnableResizing"))
	FKey DragKey;

	UPROPERTY(meta = (BindWidgetOptional))
	class UBorder* WindowTitleBorderWidget;

	UPROPERTY(meta = (BindWidgetOptional))
	class UCanvasPanel* ChildWidgetCanvas;

	UPROPERTY(meta = (BindWidgetOptional))
	class UBorder* ResizeAreaWidget;

	UPROPERTY()
	class UCanvasPanelSlot* ParentSlot;

	UPROPERTY()
	class UUserWidget* ChildWidget;

	uint8 bIsMouseButtonDown : 1;
	uint8 bIsDragging : 1;
	uint8 bIsResizing : 1;
	uint8 bIsAlignmentAccountedFor : 1;
	uint8 bStartInCenterScreen : 1;

	FVector2D LastMousePosition;
	FVector2D PreResizeAlignment;
	FVector2D PreResizeOffset;
	FVector2D PreDragSize;

	FVector2D StartSize;

	int32 CurrentZOrder;

	/* TongSim Window */
public:
	void SetWindowTile(FText InText);

private:
	UPROPERTY(meta = (BindWidgetOptional))
	class UTSButtonBase* CloseButton;

	UPROPERTY(meta = (BindWidgetOptional))
	class UTextBlock* WindowTitle;
};
