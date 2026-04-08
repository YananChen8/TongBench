// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UI/TSWidgetBase.h"
#include "TSButtonBase.generated.h"

class UButton;

/* ---- All properties must be EditDefaultsOnly, BlueprintReadOnly !!! -----
 *       we return the CDO to blueprints, so we cannot allow any changes (blueprint doesn't support const variables)
 */
UCLASS(Abstract, Blueprintable, ClassGroup = UI, meta = (Category = "TongSim|UI"))
class TONGSIMGAMEPLAY_API UTSButtonStyle : public UObject
{
	GENERATED_BODY()

	virtual bool NeedsLoadForServer() const override;

public:
	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly)
	FSlateBrush Base;

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly)
	FSlateBrush Hovered;

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly)
	FSlateBrush Pressed;

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly)
	FSlateBrush Disabled;

	/** The button content padding to apply for each size */
	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "Properties")
	FMargin ButtonPadding;

	/** The sound to play when the button is pressed */
	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "Properties", meta = (DisplayName = "Pressed Sound"))
	FSlateSound PressedSlateSound;

	/** The sound to play when the button is hovered */
	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "Properties", meta = (DisplayName = "Hovered Sound"))
	FSlateSound HoveredSlateSound;
};


/**
 *
 */
UCLASS(Abstract, BlueprintType, Blueprintable, ClassGroup = UI, meta = (Category = "TongSim|UI", DisableNativeTick))
class TONGSIMGAMEPLAY_API UTSButtonBase : public UTSWidgetBase
{
	GENERATED_BODY()
	/**
	 * Root UButton:
	 */
public:
	virtual bool Initialize() override;

protected:
	virtual UButton* ConstructInternalButton();
	virtual void SynchronizeProperties() override;

private:
	TWeakObjectPtr<UButton> RootButton;

	/**
	 * Root Button Style:
	 */
protected:
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = Style, meta = (ExposeOnSpawn = true))
	TSubclassOf<UTSButtonStyle> Style;

private:
	// TODO: Add state control like CommonUI?
	void RefreshRootButtonStyle();
	const UTSButtonStyle* GetStyleCDO() const;

	UPROPERTY()
	FButtonStyle RootButtonStyle;

	/**
	 * TS Button
	 */
public:
	UFUNCTION(BlueprintCallable)
	void SetButtonText(const FText& InText);

	// TODO: Expose setter to blueprint ?
	virtual void SetIsEnabled(bool bInIsEnabled) override;


	DECLARE_EVENT(UTSButtonBase, FTongSimButtonEvent);
	FTongSimButtonEvent& OnClicked() const { return OnClickedEvent; }

private:
	mutable FTongSimButtonEvent OnClickedEvent;

protected:
	virtual void NativePreConstruct() override;

	// TODO: Add event like FCommonButtonEvent?
	UFUNCTION(BlueprintImplementableEvent, Category = "TongSim|UI", meta = (DisplayName = "On Hovered"))
	void BP_OnHovered();
	UFUNCTION()
	virtual void NativeOnHovered();

	UFUNCTION(BlueprintImplementableEvent, Category = "TongSim|UI", meta = (DisplayName = "On Unhovered"))
	void BP_OnUnhovered();
	UFUNCTION()
	virtual void NativeOnUnhovered();

	UFUNCTION(BlueprintImplementableEvent, Category = "TongSim|UI", meta = (DisplayName = "On Pressed"))
	void BP_OnPressed();
	UFUNCTION()
	virtual void NativeOnPressed();

	UFUNCTION(BlueprintImplementableEvent, Category = "TongSim|UI", meta = (DisplayName = "On Released"))
	void BP_OnReleased();
	UFUNCTION()
	virtual void NativeOnReleased();

	UFUNCTION(BlueprintImplementableEvent, Category = "TongSim|UI", meta = (DisplayName = "On Clicked"))
	void BP_OnClicked();
	UFUNCTION()
	virtual void NativeOnClicked();

	UFUNCTION(BlueprintImplementableEvent, Category = "TongSim|UI", meta = (DisplayName = "On Enabled"))
	void BP_OnEnabled();
	virtual void NativeOnEnabled();

	UFUNCTION(BlueprintImplementableEvent, Category = "TongSim|UI", meta = (DisplayName = "On Disabled"))
	void BP_OnDisabled();
	virtual void NativeOnDisabled();

	UFUNCTION(BlueprintImplementableEvent, Category = "TongSim|UI", meta = (DisplayName = "On Update Button Text"))
	void BP_OnUpdateButtonText(const FText& InText);
	virtual void NativeOnUpdateButtonText(const FText& InText);

private:
	UPROPERTY(EditAnywhere, Category="TS|UI", meta=(InlineEditConditionToggle))
	bool bOverride_ButtonText = false;

	UPROPERTY(EditAnywhere, Category="TS|UI", meta=(EditCondition="bOverride_ButtonText"))
	FText TSButtonText;

	void RefreshButtonText();

	/** True if this button is currently enabled */
	uint8 bButtonEnabled:1;

	void EnableButton();
	void DisableButton();
};
