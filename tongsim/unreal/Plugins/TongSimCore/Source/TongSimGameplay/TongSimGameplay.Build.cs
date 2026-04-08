using UnrealBuildTool;

public class TongSimGameplay : ModuleRules
{
    public TongSimGameplay(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicIncludePaths.Add(ModuleDirectory);

        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
                "NetCore",
                "PhysicsCore",
                "OnlineSubsystem",
                "HeadMountedDisplay",
                "ModularGameplay",
                "GameplayTags",
                "EnhancedInput",
                "ModularGameplayActors"
            }
        );

        PrivateDependencyModuleNames.AddRange(
            new string[]
            {
                "CoreUObject",
                "Engine",
                "Slate",
                "SlateCore",
                "InputCore",
                "UMG",
                "OnlineSubsystemUtils",
                "DeveloperSettings",
                "Sockets",
                "AIModule"
            }
        );
    }
}
