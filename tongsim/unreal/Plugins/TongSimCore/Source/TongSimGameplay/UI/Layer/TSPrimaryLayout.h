// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameplayTagContainer.h"
#include "TSActivatableWidget.h"
#include "TSActivatableWidgetContainerBase.h"
#include "Engine/AssetManager.h"
#include "Engine/StreamableManager.h"
#include "UI/TSWidgetBase.h"
#include "TSPrimaryLayout.generated.h"

/**
 * The state of an async load operation for the UI.
 */
enum class EAsyncWidgetLayerState : uint8
{
	Canceled,
	Initialize,
	AfterPush
};

class UTSActivatableWidgetContainerBase;

/**
 *
 */
UCLASS(Abstract, meta = (DisableNativeTick))
class TONGSIMGAMEPLAY_API UTSPrimaryLayout : public UTSWidgetBase
{
	GENERATED_BODY()
public:
	UTSPrimaryLayout(const FObjectInitializer& ObjectInitializer);

protected:
	/** Register a layer that widgets can be pushed onto. */
	UFUNCTION(BlueprintCallable, Category="Layer")
	void RegisterLayer(UPARAM(meta = (Categories = "UI.Layer")) FGameplayTag LayerTag, UTSActivatableWidgetContainerBase* LayerWidget);

	// Find the widget if it exists on any of the layers and remove it from the layer.
	void FindAndRemoveWidgetFromLayer(UTSActivatableWidget* ActivatableWidget);

	// Get the layer widget for the given layer tag.
	UTSActivatableWidgetContainerBase* GetLayerWidget(FGameplayTag LayerName);

private:
	UPROPERTY(Transient, meta = (Categories = "UI.Layer"))
	TMap<FGameplayTag, TObjectPtr<UTSActivatableWidgetContainerBase>> Layers;

	/**
	 * Push Widget to Layer with Async or Sync.
	 */
public:

	template <typename ActivatableWidgetT = UTSActivatableWidget>
	TSharedPtr<FStreamableHandle> PushWidgetToLayerStackAsync(FGameplayTag LayerName, TSoftClassPtr<UTSActivatableWidget> ActivatableWidgetClass)
	{
		return PushWidgetToLayerStackAsync<ActivatableWidgetT>(LayerName, ActivatableWidgetClass,
			[](EAsyncWidgetLayerState, ActivatableWidgetT*) {});
	}

	template <typename ActivatableWidgetT = UTSActivatableWidget>
	TSharedPtr<FStreamableHandle> PushWidgetToLayerStackAsync(FGameplayTag LayerName, TSoftClassPtr<UTSActivatableWidget> ActivatableWidgetClass,
		TFunction<void(EAsyncWidgetLayerState, ActivatableWidgetT*)> StateFunc)
	{
		static_assert(TIsDerivedFrom<ActivatableWidgetT, UTSActivatableWidget>::IsDerived, "Only ActivatableWidgets can be used here");

		FStreamableManager& StreamableManager = UAssetManager::Get().GetStreamableManager();
		TSharedPtr<FStreamableHandle> StreamingHandle = StreamableManager.RequestAsyncLoad(ActivatableWidgetClass.ToSoftObjectPath(),
			FStreamableDelegate::CreateWeakLambda(this,[this, LayerName, ActivatableWidgetClass, StateFunc]()
			{
				ActivatableWidgetT* Widget = PushWidgetToLayerStack<ActivatableWidgetT>(LayerName, ActivatableWidgetClass.Get(), [StateFunc](ActivatableWidgetT& WidgetToInit) {
					StateFunc(EAsyncWidgetLayerState::Initialize, &WidgetToInit);
				});

				StateFunc(EAsyncWidgetLayerState::AfterPush, Widget);
			})
		);

		// Setup a cancel delegate so that we can resume input if this handler is canceled.
		StreamingHandle->BindCancelDelegate(FStreamableDelegate::CreateWeakLambda(this,
			[this, StateFunc]()
			{
				StateFunc(EAsyncWidgetLayerState::Canceled, nullptr);
			})
		);

		return StreamingHandle;
	}

	template <typename ActivatableWidgetT = UTSActivatableWidget>
	ActivatableWidgetT* PushWidgetToLayerStack(FGameplayTag LayerName, UClass* ActivatableWidgetClass)
	{
		return PushWidgetToLayerStack<ActivatableWidgetT>(LayerName, ActivatableWidgetClass, [](ActivatableWidgetT&) {});
	}

	template <typename ActivatableWidgetT = UTSActivatableWidget>
	ActivatableWidgetT* PushWidgetToLayerStack(FGameplayTag LayerName, UClass* ActivatableWidgetClass, TFunctionRef<void(ActivatableWidgetT&)> InitInstanceFunc)
	{
		static_assert(TIsDerivedFrom<ActivatableWidgetT, UTSActivatableWidget>::IsDerived, "Only ActivatableWidgets can be used here");

		if (UTSActivatableWidgetContainerBase* Layer = GetLayerWidget(LayerName))
		{
			return Layer->AddWidget<ActivatableWidgetT>(ActivatableWidgetClass, InitInstanceFunc);
		}

		return nullptr;
	}
};
