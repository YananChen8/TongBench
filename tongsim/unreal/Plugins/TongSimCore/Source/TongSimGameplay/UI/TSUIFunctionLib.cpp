// Fill out your copyright notice in the Description page of Project Settings.


#include "TSUIFunctionLib.h"
#include "TSUISubsystem.h"
#include "Layer/TSPrimaryLayout.h"
#include "TSLogChannels.h"

UTSActivatableWidget* UTSUIFunctionLib::PushWidgetToLayerForPlayer(TSubclassOf<UTSActivatableWidget> WidgetClass, FGameplayTag LayerName)
{
	if (UTSUISubsystem* UISubsystem = UTSUISubsystem::GetInstance())
	{
		if (WidgetClass == nullptr)
		{
			UE_LOG(LogTongSimCore, Error, TEXT("Push widget to layer get null widget class."));
			return nullptr;
		}

		if (UTSPrimaryLayout* RootLayout = UISubsystem->GetCurrentPrimaryLayout())
		{
			return RootLayout->PushWidgetToLayerStack<UTSActivatableWidget>(LayerName, WidgetClass);
		}
	}

	return nullptr;
}
