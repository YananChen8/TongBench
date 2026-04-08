// Fill out your copyright notice in the Description page of Project Settings.


#include "TSWidgetBase.h"

TSharedPtr<FSlateUser> UTSWidgetBase::GetOwnerSlateUser() const
{
	ULocalPlayer* LocalPlayer = GetOwningLocalPlayer();
	return LocalPlayer ? LocalPlayer->GetSlateUser() : nullptr;
}
