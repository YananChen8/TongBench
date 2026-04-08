// Fill out your copyright notice in the Description page of Project Settings.


#include "TSActivatableWidgetContainerBase.h"

#include "TSLogChannels.h"
#include "Widgets/Layout/SSpacer.h"
#include "Widgets/Layout/SWidgetSwitcher.h"

namespace TSActivatableWidgetContainerHelper
{
	UTSActivatableWidget* ActivatableWidgetFromSlate(const TSharedPtr<SWidget>& SlateWidget)
	{
		if (SlateWidget && SlateWidget != SNullWidget::NullWidget && ensure(SlateWidget->GetType().IsEqual(TEXT("SObjectWidget"))))
		{
			UTSActivatableWidget* ActivatableWidget = Cast<UTSActivatableWidget>(StaticCastSharedPtr<SObjectWidget>(SlateWidget)->GetWidgetObject());
			if (ensure(ActivatableWidget))
			{
				return ActivatableWidget;
			}
		}
		return nullptr;
	}
}

UTSActivatableWidgetContainerBase::UTSActivatableWidgetContainerBase(const FObjectInitializer& Initializer) : Super(Initializer) , GeneratedWidgetsPool(*this)
{
	SetVisibilityInternal(ESlateVisibility::Collapsed);
}

UTSActivatableWidget* UTSActivatableWidgetContainerBase::GetActiveWidget() const
{
	return MySwitcher ? TSActivatableWidgetContainerHelper::ActivatableWidgetFromSlate(MySwitcher->GetActiveWidget()) : nullptr;
}

int32 UTSActivatableWidgetContainerBase::GetNumWidgets() const
{
	return WidgetList.Num();
}

void UTSActivatableWidgetContainerBase::ClearWidgets()
{
	SetSwitcherIndex(0);
}

TSharedRef<SWidget> UTSActivatableWidgetContainerBase::RebuildWidget()
{
	MyOverlay = SNew(SOverlay)
	+ SOverlay::Slot()
	[
		SAssignNew(MySwitcher, SWidgetSwitcher)
	];
	// this intend to intercept all input when switcher is transitioning
	//
	// + SOverlay::Slot()
	// [
	// 	SAssignNew(MyInputGuard, SSpacer)
	// 	.Visibility(EVisibility::Collapsed)
	// ];

	// We always want a 0th slot to be able to animate the first real entry in and out
	MySwitcher->AddSlot()[SNullWidget::NullWidget];

	return MyOverlay.ToSharedRef();
}

void UTSActivatableWidgetContainerBase::ReleaseSlateResources(bool bReleaseChildren)
{
	Super::ReleaseSlateResources(bReleaseChildren);

	MyOverlay.Reset();
	MySwitcher.Reset();
	ReleasedWidgets.Empty();
	GeneratedWidgetsPool.ReleaseAllSlateResources();
}

void UTSActivatableWidgetContainerBase::OnWidgetRebuilt()
{
	Super::OnWidgetRebuilt();

	if (!IsDesignTime())
	{
		// When initially created, fake that we just did an initial transition to index 0
		HandleActiveIndexChanged(0);
	}
}

void UTSActivatableWidgetContainerBase::RemoveWidget(UTSActivatableWidget& WidgetToRemove)
{
	if (&WidgetToRemove == GetActiveWidget())
	{
		// To remove the active widget, just deactivate it (if it's already deactivated, then we're already in the process of ditching it)
		if (WidgetToRemove.IsActivated())
		{
			WidgetToRemove.DeactivateWidget();
		}
		else
		{
			bRemoveDisplayedWidgetPostTransition = true;
		}
	}
	else
	{
		// Otherwise if the widget isn't actually being shown right now, yank it right on out
		TSharedPtr<SWidget> CachedWidget = WidgetToRemove.GetCachedWidget();
		if (CachedWidget && MySwitcher)
		{
			ReleaseWidget(CachedWidget.ToSharedRef());
		}
	}
}

void UTSActivatableWidgetContainerBase::SetSwitcherIndex(int32 TargetIndex)
{
	if (MySwitcher && MySwitcher->GetActiveWidgetIndex() != TargetIndex)
	{
		if (DisplayedWidget)
		{
			DisplayedWidget->OnDeactivated().RemoveAll(this);
			if (DisplayedWidget->IsActivated())
			{
				DisplayedWidget->DeactivateWidget();
			}
			else if (MySwitcher->GetActiveWidgetIndex() != 0)
			{
				// The displayed widget has already been deactivated by something other than us, so it should be removed from the container
				// We still need it to remain briefly though until we transition to the new index - then we can remove this entry's slot
				bRemoveDisplayedWidgetPostTransition = true;
			}
		}

		MySwitcher->SetActiveWidgetIndex(TargetIndex);
		// TODO: Here is different from common ui, Find a way to bind this handle to a delegate like OnActiveIndexChanged in MySwitcher?
		HandleActiveIndexChanged(TargetIndex);
	}
}

UTSActivatableWidget* UTSActivatableWidgetContainerBase::AddWidgetInternal(TSubclassOf<UTSActivatableWidget> ActivatableWidgetClass, TFunctionRef<void(UTSActivatableWidget&)> InitFunc)
{
	if (UTSActivatableWidget* WidgetInstance = GeneratedWidgetsPool.GetOrCreateInstance(ActivatableWidgetClass))
	{
		InitFunc(*WidgetInstance);
		RegisterInstanceInternal(*WidgetInstance);
		return WidgetInstance;
	}
	return nullptr;
}

void UTSActivatableWidgetContainerBase::RegisterInstanceInternal(UTSActivatableWidget& InWidget)
{
	if (ensure(!WidgetList.Contains(&InWidget)))
	{
		WidgetList.Add(&InWidget);
		OnWidgetAddedToList(InWidget);
	}
}

void UTSActivatableWidgetContainerBase::HandleActiveIndexChanged(int32 ActiveWidgetIndex)
{
	// Remove all slots above the currently active one and release the widgets back to the pool
	while (MySwitcher->GetNumWidgets() - 1 > ActiveWidgetIndex)
	{
		TSharedPtr<SWidget> WidgetToRelease = MySwitcher->GetWidget(MySwitcher->GetNumWidgets() - 1);
		if (ensure(WidgetToRelease))
		{
			ReleaseWidget(WidgetToRelease.ToSharedRef());
		}
	}

	// Also remove the widget that we just transitioned away from if desired
	if (DisplayedWidget && bRemoveDisplayedWidgetPostTransition)
	{
		if (TSharedPtr<SWidget> DisplayedSlateWidget = DisplayedWidget->GetCachedWidget())
		{
			ReleaseWidget(DisplayedSlateWidget.ToSharedRef());
		}
	}
	bRemoveDisplayedWidgetPostTransition = false;

	// Activate the widget that's now being displayed
	DisplayedWidget = TSActivatableWidgetContainerHelper::ActivatableWidgetFromSlate(MySwitcher->GetActiveWidget());
	if (DisplayedWidget)
	{
		SetVisibility(ESlateVisibility::SelfHitTestInvisible);

		DisplayedWidget->OnDeactivated().AddUObject(this, &ThisClass::HandleActiveWidgetDeactivated, ToRawPtr(DisplayedWidget));
		DisplayedWidget->ActivateWidget();

		if (UWorld* MyWorld = GetWorld())
		{
			FTimerManager& TimerManager = MyWorld->GetTimerManager();
			TimerManager.SetTimerForNextTick(FTimerDelegate::CreateWeakLambda(this, [this]() { InvalidateLayoutAndVolatility(); }));
		}
	}
	else
	{
		SetVisibility(ESlateVisibility::Collapsed);
	}
}

void UTSActivatableWidgetContainerBase::HandleActiveWidgetDeactivated(UTSActivatableWidget* DeactivatedWidget)
{
	// When the currently displayed widget deactivates, transition the switcher to the preceding slot (if it exists)
	// We'll clean up this slot once the switcher index actually changes
	if (ensure(DeactivatedWidget == DisplayedWidget) && MySwitcher && MySwitcher->GetActiveWidgetIndex() > 0)
	{
		DisplayedWidget->OnDeactivated().RemoveAll(this);
		MySwitcher->SetActiveWidgetIndex(MySwitcher->GetActiveWidgetIndex() - 1);
	}
}

void UTSActivatableWidgetContainerBase::ReleaseWidget(const TSharedRef<SWidget>& WidgetToRelease)
{
	if (UTSActivatableWidget* ActivatableWidget = TSActivatableWidgetContainerHelper::ActivatableWidgetFromSlate(WidgetToRelease))
	{
		GeneratedWidgetsPool.Release(ActivatableWidget, true);
		WidgetList.Remove(ActivatableWidget);
	}

	if (MySwitcher->RemoveSlot(WidgetToRelease) != INDEX_NONE)
	{
		ReleasedWidgets.Add(WidgetToRelease);
		if (ReleasedWidgets.Num() == 1)
		{
			FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateWeakLambda(this,
				[this](float)
				{
					QUICK_SCOPE_CYCLE_COUNTER(STAT_UCommonActivatableWidgetContainerBase_ReleaseWidget);
					ReleasedWidgets.Reset();
					return false;
				}));
		}
	}
}


void UTSActivatableWidgetContainerStack::OnWidgetAddedToList(UTSActivatableWidget& AddedWidget)
{
	// Here is implementation of add a widget to layer:
	if (MySwitcher)
	{
		MySwitcher->AddSlot()[AddedWidget.TakeWidget()]; // Refer to TPanelChildren in Children.h, Search "operator[](const TSharedRef<SWidget>& InChildWidget)".
		SetSwitcherIndex(MySwitcher->GetNumWidgets() - 1);
	}
}

void UTSActivatableWidgetContainerQueue::OnWidgetAddedToList(UTSActivatableWidget& AddedWidget)
{
	UE_LOG(LogTongSimCore, Fatal, TEXT("Called activatable widget container queue which has not been implementated."));
}
