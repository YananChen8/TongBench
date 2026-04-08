// Fill out your copyright notice in the Description page of Project Settings.


#include "TSPrimaryLayout.h"

UTSPrimaryLayout::UTSPrimaryLayout(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer)
{
}

void UTSPrimaryLayout::RegisterLayer(FGameplayTag LayerTag, UTSActivatableWidgetContainerBase* LayerWidget)
{
	if (!IsDesignTime())
	{
		Layers.Add(LayerTag, LayerWidget);
	}
}

void UTSPrimaryLayout::FindAndRemoveWidgetFromLayer(UTSActivatableWidget* ActivatableWidget)
{
	for (const auto& Layer : Layers)
	{
		Layer.Value->RemoveWidget(*ActivatableWidget);
	}
}

UTSActivatableWidgetContainerBase* UTSPrimaryLayout::GetLayerWidget(FGameplayTag LayerName)
{
	return Layers.FindRef(LayerName);
}
