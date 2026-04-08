// Fill out your copyright notice in the Description page of Project Settings.


#include "TSButtonBase.h"

#include "Blueprint/WidgetTree.h"
#include "Components/Button.h"
#include "Components/ButtonSlot.h"
#include "Engine/UserInterfaceSettings.h"

// TODO: Open this with 5.4
// #include "Binding/States/WidgetStateRegistration.h"

bool UTSButtonStyle::NeedsLoadForServer() const
{
	return GetDefault<UUserInterfaceSettings>()->bLoadWidgetsOnDedicatedServer;
}

bool UTSButtonBase::Initialize()
{
	const bool bInitializedThisCall = Super::Initialize();

	if (bInitializedThisCall)
	{
		UButton* RootButtonRaw = ConstructInternalButton();
		check(IsValid(RootButtonRaw));
		RootButton = RootButtonRaw;
		if (WidgetTree->RootWidget)
		{
			UButtonSlot* NewSlot = Cast<UButtonSlot>(RootButton->AddChild(WidgetTree->RootWidget));
			NewSlot->SetPadding(FMargin());
			NewSlot->SetHorizontalAlignment(EHorizontalAlignment::HAlign_Fill);
			NewSlot->SetVerticalAlignment(EVerticalAlignment::VAlign_Fill);
			WidgetTree->RootWidget = RootButtonRaw;

			RootButton->OnClicked.AddUniqueDynamic(this, &ThisClass::NativeOnClicked);
			RootButton->OnHovered.AddUniqueDynamic(this, &ThisClass::NativeOnHovered);
			RootButton->OnUnhovered.AddUniqueDynamic(this, &ThisClass::NativeOnUnhovered);
			RootButton->OnPressed.AddUniqueDynamic(this, &ThisClass::NativeOnPressed);
			RootButton->OnReleased.AddUniqueDynamic(this, &ThisClass::NativeOnReleased);
		}
	}
	return bInitializedThisCall;
}

UButton* UTSButtonBase::ConstructInternalButton()
{
	return WidgetTree->ConstructWidget<UButton>(UButton::StaticClass(), FName(TEXT("InternalRootButtonBase")));
}

void UTSButtonBase::SynchronizeProperties()
{
	Super::SynchronizeProperties();
	RefreshRootButtonStyle();
}

void UTSButtonBase::RefreshRootButtonStyle()
{
	{
		const UTSButtonStyle* ButtonStyle = GetStyleCDO();
		UButton* RootButtonPtr = RootButton.Get();
		if (ButtonStyle && RootButtonPtr)
		{
			RootButtonStyle.Normal = ButtonStyle->Base;
			RootButtonStyle.Hovered = ButtonStyle->Hovered;
			RootButtonStyle.Pressed = ButtonStyle->Pressed;
			RootButtonStyle.Disabled = ButtonStyle->Disabled;

			RootButtonStyle.NormalPadding = ButtonStyle->ButtonPadding;
			RootButtonStyle.PressedPadding = ButtonStyle->ButtonPadding;

			RootButtonStyle.PressedSlateSound = ButtonStyle->PressedSlateSound;
			RootButtonStyle.HoveredSlateSound = ButtonStyle->HoveredSlateSound;

			RootButtonPtr->SetStyle(RootButtonStyle);
		}
	}
}

const UTSButtonStyle* UTSButtonBase::GetStyleCDO() const
{
		if (Style)
		{
			// if (const UTSButtonStyle* ButtonStyle = Cast<UTSButtonStyle>(Style->ClassDefaultObject))
			if (const UTSButtonStyle* ButtonStyle = GetDefault<UTSButtonStyle>(Style))
			{
				return ButtonStyle;
			}
		}
		return nullptr;
}

void UTSButtonBase::SetButtonText(const FText& InText)
{
	bOverride_ButtonText = !InText.IsEmpty();
	TSButtonText = InText;
	RefreshButtonText();
}

void UTSButtonBase::SetIsEnabled(bool bInIsEnabled)
{
	bool bValueChanged = bButtonEnabled != bInIsEnabled;

	if (bInIsEnabled)
	{
		// TGuardValue<bool> StateBroadcastGuard(bShouldBroadcastState, false);
		Super::SetIsEnabled(bInIsEnabled);
		EnableButton();
	}
	else
	{
		// Change the underlying enabled bool but do not call the case because we don't want to propogate it to the underlying SWidget
		// TGuardValue<bool> StateBroadcastGuard(bShouldBroadcastState, false);
		Super::SetIsEnabled(bInIsEnabled);
		DisableButton();
	}

	/*if (bValueChanged)
	{
		// Note: State is disabled, so we broadcast !bIsEnabled
		BroadcastBinaryPostStateChange(UWidgetDisabledStateRegistration::Bit, !bInIsEnabled);
	}*/
}

void UTSButtonBase::NativePreConstruct()
{
	Super::NativePreConstruct();
	RefreshButtonText();
}

void UTSButtonBase::NativeOnHovered()
{
	BP_OnHovered();
	// BroadcastBinaryPostStateChange(UWidgetHoveredStateRegistration::Bit, true);
}

void UTSButtonBase::NativeOnUnhovered()
{
	BP_OnUnhovered();
	// BroadcastBinaryPostStateChange(UWidgetHoveredStateRegistration::Bit, false);
}

void UTSButtonBase::NativeOnPressed()
{
	BP_OnPressed();
	// BroadcastBinaryPostStateChange(UWidgetPressedStateRegistration::Bit, true);
}

void UTSButtonBase::NativeOnReleased()
{
	BP_OnPressed();
	// BroadcastBinaryPostStateChange(UWidgetPressedStateRegistration::Bit, false);
}

void UTSButtonBase::NativeOnClicked()
{
	BP_OnClicked();
	OnClicked().Broadcast();
}

void UTSButtonBase::NativeOnEnabled()
{
	BP_OnEnabled();
}

void UTSButtonBase::NativeOnDisabled()
{
	BP_OnDisabled();
}

void UTSButtonBase::NativeOnUpdateButtonText(const FText& InText)
{
	BP_OnUpdateButtonText(InText);
}

void UTSButtonBase::RefreshButtonText()
{
	if (bOverride_ButtonText)
	{
		NativeOnUpdateButtonText(TSButtonText);
	}
}

void UTSButtonBase::EnableButton()
{
	if (!bButtonEnabled)
	{
		bButtonEnabled = true;
		NativeOnEnabled();
	}
}

void UTSButtonBase::DisableButton()
{
	if (bButtonEnabled)
	{
		bButtonEnabled = false;
		NativeOnDisabled();
	}
}
