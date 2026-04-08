#include "TTDebugTickFunctionBase.h"
#include "TTBaseDebugger.h"

void FTTDebugTickFunc_PrePhysics::ExecuteTick(float DeltaTime, ELevelTick TickType, ENamedThreads::Type CurrentThread, const FGraphEventRef& MyCompletionGraphEvent)
{
	if (DebuggerRawPtr && IsValidChecked(DebuggerRawPtr) && !DebuggerRawPtr->IsUnreachable())
	{
		if (TickType != LEVELTICK_ViewportsOnly || DebuggerRawPtr->ShouldTickIfViewportsOnly())
		{
			DebuggerRawPtr->PrePhysicsTickActor(DeltaTime*DebuggerRawPtr->CustomTimeDilation, TickType, *this);
		}
	}
}

void FTTDebugTickFunc_DuringPhysics::ExecuteTick(float DeltaTime, ELevelTick TickType, ENamedThreads::Type CurrentThread, const FGraphEventRef& MyCompletionGraphEvent)
{
	if (DebuggerRawPtr && IsValidChecked(DebuggerRawPtr) && !DebuggerRawPtr->IsUnreachable())
	{
		if (TickType != LEVELTICK_ViewportsOnly || DebuggerRawPtr->ShouldTickIfViewportsOnly())
		{
			DebuggerRawPtr->DuringPhysicsTickActor(DeltaTime*DebuggerRawPtr->CustomTimeDilation, TickType, *this);
		}
	}
}

void FTTDebugTickFunc_PostPhysics::ExecuteTick(float DeltaTime, ELevelTick TickType, ENamedThreads::Type CurrentThread, const FGraphEventRef& MyCompletionGraphEvent)
{
	if (DebuggerRawPtr && IsValidChecked(DebuggerRawPtr) && !DebuggerRawPtr->IsUnreachable())
	{
		if (TickType != LEVELTICK_ViewportsOnly || DebuggerRawPtr->ShouldTickIfViewportsOnly())
		{
			DebuggerRawPtr->PostPhysicsTickActor(DeltaTime*DebuggerRawPtr->CustomTimeDilation, TickType, *this);
		}
	}
}

void FTTDebugTickFunc_PostUpdateWork::ExecuteTick(float DeltaTime, ELevelTick TickType, ENamedThreads::Type CurrentThread, const FGraphEventRef& MyCompletionGraphEvent)
{
	if (DebuggerRawPtr && IsValidChecked(DebuggerRawPtr) && !DebuggerRawPtr->IsUnreachable())
	{
		if (TickType != LEVELTICK_ViewportsOnly || DebuggerRawPtr->ShouldTickIfViewportsOnly())
		{
			DebuggerRawPtr->PostUpdateWorkTickActor(DeltaTime*DebuggerRawPtr->CustomTimeDilation, TickType, *this);
		}
	}
}
