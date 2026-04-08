#include "TSHUDBase.h"

#include "TSUIFunctionLib.h"
#include "TSGameplayTags.h"
#include "Common/TSWindowBase.h"
#include "TSLogChannels.h"
#include "Debug/TSDebugPanelWidget.h"


ATSHUDBase::ATSHUDBase()
{
	PrimaryActorTick.bCanEverTick = false;
}

UTSWindowBase* ATSHUDBase::CreateDebugWindowWithChildWidget(TSubclassOf<UUserWidget> ChildWidgetClass, const FVector2D StartSize, const FText& WindowTitle, const bool bCenterToScreen, const bool bCanDrag, const bool bCanResize)
{
	if (DebugPanelWidget.IsValid() && DebugWindowClass)
	{
		if (UTSWindowBase* DebugWindow = DebugPanelWidget->AddNewDebugWindow(DebugWindowClass, ChildWidgetClass, StartSize, bCenterToScreen, bCanDrag, bCanResize))
		{
			DebugWindow->SetWindowTile(WindowTitle);
		}
	}
	return nullptr;
}

void ATSHUDBase::BeginPlay()
{
	Super::BeginPlay();

	// TODO by wukunlun: Add debug condition check and macro?
	InitDebugPanel();
}

void ATSHUDBase::InitDebugPanel()
{
	if (DebugPanelWidgetClass)
	{
		DebugPanelWidget = Cast<UTSDebugPanelWidget>(UTSUIFunctionLib::PushWidgetToLayerForPlayer(DebugPanelWidgetClass, TongSimGameplayTags::UI_Layer_Debug));
		if (DebugPanelWidget.IsValid())
		{
			UE_LOG(LogTongSimCore, Log, TEXT("Init Debug Canvas Panel Widget"));
		}
	}
}
