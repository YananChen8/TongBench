// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "TSActivatableWidget.h"
#include "Blueprint/UserWidgetPool.h"
#include "Components/Widget.h"
#include "TSActivatableWidgetContainerBase.generated.h"

/**
 *  refer to common ui.
 */
UCLASS(Abstract)
class TONGSIMGAMEPLAY_API UTSActivatableWidgetContainerBase : public UWidget
{
	GENERATED_BODY()
public:
	UTSActivatableWidgetContainerBase(const FObjectInitializer& Initializer);

	UFUNCTION(BlueprintCallable)
	UTSActivatableWidget* GetActiveWidget() const;

	const TArray<UTSActivatableWidget*>& GetWidgetList() const { return WidgetList; }

	int32 GetNumWidgets() const;

	UFUNCTION(BlueprintCallable)
	void ClearWidgets();

protected:
	virtual TSharedRef<SWidget> RebuildWidget() override;
	virtual void ReleaseSlateResources(bool bReleaseChildren) override;
	virtual void OnWidgetRebuilt() override;

	UPROPERTY(Transient)
	TArray<TObjectPtr<UTSActivatableWidget>> WidgetList;

	UPROPERTY(Transient)
	TObjectPtr<UTSActivatableWidget> DisplayedWidget;

	UPROPERTY(Transient)
	FUserWidgetPool GeneratedWidgetsPool;

	TSharedPtr<SOverlay> MyOverlay;
	TSharedPtr<SWidgetSwitcher> MySwitcher;

private:
	bool bRemoveDisplayedWidgetPostTransition = false;

	/*
	 * Add Widget implementation func:
	 */
public:
	void RemoveWidget(UTSActivatableWidget& WidgetToRemove);

protected:
	virtual void OnWidgetAddedToList(UTSActivatableWidget& AddedWidget) { unimplemented(); }
	void SetSwitcherIndex(int32 TargetIndex);

private:
	UTSActivatableWidget* AddWidgetInternal(TSubclassOf<UTSActivatableWidget> ActivatableWidgetClass, TFunctionRef<void (UTSActivatableWidget&)> InitFunc);
	void RegisterInstanceInternal(UTSActivatableWidget& InWidget);

	void HandleActiveIndexChanged(int32 ActiveWidgetIndex);
	void HandleActiveWidgetDeactivated(UTSActivatableWidget* DeactivatedWidget);

	/**
	 *  Copy from CommonUI, Key : We hold a reference to widget for more one frame to ensure HittestGrid is working correctly
	 *
	 *
	 * This is a bit hairy and very edge-casey, but a necessary measure to ensure expected Slate interaction behavior.
	 *
	 * Since we immediately remove slots from our switcher in response to changes to the active index of the switcher, we can
	 * wind up confusing the HittestGrid for that frame. The grid (correctly) thinks the widget at the previously displayed index is what it
	 * should interact with, but it skips it because we've already released all references and destroyed it. This causes pointer
	 * input (most importantly the synthetic mouse move) to fall through our container for that frame, potentially triggering interactions
	 * with elements that, as far as any user can tell, were never actually visible!
	 *
	 * So, when we remove a slot, we hold a reference to the SWidget in that slot for a single frame, to ensure hittest grid integrity.
	 * This does delay destruction of the removed SObjectWidget by one frame, but that does not present any discernable issue,
	 * as it's no different from any other inactive widget within a switcher.
	 */
	void ReleaseWidget(const TSharedRef<SWidget>& WidgetToRelease);
	TArray<TSharedPtr<SWidget>> ReleasedWidgets;

	/*
	 * Add Widget Interface
	 */
public:
	template <typename ActivatableWidgetT = UTSActivatableWidget>
	ActivatableWidgetT* AddWidget(TSubclassOf<UTSActivatableWidget> ActivatableWidgetClass)
	{
		// Don't actually add the widget if the cast will fail
		if (ActivatableWidgetClass && ActivatableWidgetClass->IsChildOf<ActivatableWidgetT>())
		{
			return Cast<ActivatableWidgetT>(AddWidgetInternal(ActivatableWidgetClass, [](UTSActivatableWidget&)
			{
			}));
		}
		return nullptr;
	}

	template <typename ActivatableWidgetT = UTSActivatableWidget>
	ActivatableWidgetT* AddWidget(TSubclassOf<UTSActivatableWidget> ActivatableWidgetClass, TFunctionRef<void(ActivatableWidgetT&)> InstanceInitFunc)
	{
		// Don't actually add the widget if the cast will fail
		if (ActivatableWidgetClass && ActivatableWidgetClass->IsChildOf<ActivatableWidgetT>())
		{
			return Cast<ActivatableWidgetT>(AddWidgetInternal(ActivatableWidgetClass, [&InstanceInitFunc](UTSActivatableWidget& WidgetInstance)
			{
				InstanceInitFunc(*CastChecked<ActivatableWidgetT>(&WidgetInstance));
			}));
		}
		return nullptr;
	}
};

/**
 *
 */
UCLASS()
class TONGSIMGAMEPLAY_API UTSActivatableWidgetContainerStack : public UTSActivatableWidgetContainerBase
{
	GENERATED_BODY()

protected:
	virtual void OnWidgetAddedToList(UTSActivatableWidget& AddedWidget) override;
};

/**
 *
 */
UCLASS()
class TONGSIMGAMEPLAY_API UTSActivatableWidgetContainerQueue : public UTSActivatableWidgetContainerBase
{
	GENERATED_BODY()

protected:
	virtual void OnWidgetAddedToList(UTSActivatableWidget& AddedWidget) override;
};
